
#include "main.h"
float Norm;

//Mutiple-GPU Plan Structure
typedef struct
{
    mytype *h_A, *h_B;
    float *A_normmap,*B_normmap;
    float *h_C;
    cudaStream_t stream;
} TGPUplan;

#define MATRIXOFFSETCPY(dst,src,size_row,size_col,off_row,off_col,total_col) \
for(int i=0;i<size_row;i++){ \
    for(int j=0;j<size_col;j++){ \
        dst[i][j]=GETELEMENT21(src,off_row+i,off_col+j,total_col); \
    } \
}

__global__ void unroll_get_Fnorm_pri(const float* __restrict__ A,float *A_normmap,int m,int n,int blockRowOff){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;
    int thId = threadIdx.x;
    __shared__ float sdata[LoNum*LoNum/8/32];

    int valid=0;
    const int myBlockRow = kId / (n/LoNum)+blockRowOff;
    const int myBlockCol = kId % (n/LoNum);
    const int myBlockId = myBlockRow*(n/LoNum)+myBlockCol;
    const int myThreadRow = thId / (LoNum/8);
    const int myThreadCol = thId % (LoNum/8);
    const int myFinalRow = myBlockRow*LoNum+myThreadRow;
    const int myFinalCol = myBlockCol*LoNum+myThreadCol*8;

    float val;
    valid = id > m*n? 0:1;
    if(valid){
        int tadd = myFinalRow*n+myFinalCol;
        float t1 = A[tadd];
        float t2 = A[tadd+1];
        float t3 = A[tadd+2];
        float t4 = A[tadd+3];
        float t5 = A[tadd+4];
        float t6 = A[tadd+5];
        float t7 = A[tadd+6];
        float t8 = A[tadd+7];
        val = t1*t1+t2*t2+t3*t3+t4*t4+t5*t5+t6*t6+t7*t7+t8*t8;
    } 
    
    #define FULL_MASK 0xffffffff
    for (int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }       
    if(thId%32==0){
        sdata[thId/32]=val;
    } 
    
    __syncthreads();
    float r=0;
    
    if (thId < blockDim.x/32)
    {
        val=sdata[thId];
        for (int offset = blockDim.x/32/2; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    if(thId==0){
        A_normmap[myBlockId] = sqrt(val);
    }
}

//4*32
__global__ void unroll_get_Fnorm_FP16(const half* __restrict__ A,float *A_normmap,int m,int n,int blockRowOff){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;
    int thId = threadIdx.x;
    int warpId = thId / 32;

    const int myBlockRow = kId / (n/LoNum)+blockRowOff;
    const int myBlockCol = kId % (n/LoNum);
    const int myBlockId = myBlockRow*(n/LoNum)+myBlockCol;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> chalf_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    __shared__ half sdata_half[32*32];
    __shared__ float sdata_float[4];
    
    int warpi=warpId/2;
    int warpj=warpId%2;
    
    wmma::fill_fragment(a_frag, 1.0f);
    wmma::fill_fragment(chalf_frag, 0.0f);
    wmma::load_matrix_sync(b_frag, GETOFF21(A,myBlockRow*LoNum+warpi*16,myBlockCol*LoNum+warpj*16,n), n);
    for (int i = 0; i < b_frag.num_elements; i++) {
        half t=b_frag.x[i];
        b_frag.x[i] = __float2half(__half2float(t) * __half2float(t));
    }
    wmma::mma_sync(chalf_frag, a_frag, b_frag, chalf_frag);
    
    wmma::store_matrix_sync(GETOFF21(sdata_half,warpi*16,warpj*16,32), chalf_frag, 32,wmma::mem_row_major);
    __syncthreads();
    
    wmma::load_matrix_sync(a_frag, GETOFF21(sdata_half,warpi*16,warpj*16,32), 32);
    
    
    wmma::fill_fragment(b_frag, 1.0f);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
 
    __syncthreads();
    if(thId%32==0){
        sdata_float[warpId]=c_frag.x[0];
    }
    __syncthreads();
    if(thId==0){
        A_normmap[myBlockId]=sqrt(sdata_float[0]+sdata_float[1]+sdata_float[2]+sdata_float[3]);
        
    }
}


__global__ void get_C_Threads1Element_Mul(const float* __restrict__ A,const float* __restrict__ A_normmap,const float* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset,float Norm){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;
    int thId = threadIdx.x;
    int REDUCECBL = 1<<(int)(log2(CBLMUN*1.0)+1);
    int warpId = thId / 32;
    int warpi = thId % 32;
    float norm_mul;
    const int first16 = 1-warpId/16;

    const int myBlockRow = kId / (N/LoNum) + main_row_offset; 
    const int myBlockCol = kId % (N/LoNum);
    const int myBlockRowOff = myBlockRow*LoNum;
    const int myBlockColOff = myBlockCol*LoNum;

    __shared__ int sC_bitmap[CBLMUN*2];
    __shared__ int sC_offset[CBLMUN];
    __shared__ float sA0[LoNum*LoNum],sB0[LoNum*LoNum];
    __shared__ float sA1[LoNum*LoNum],sB1[LoNum*LoNum]; 

    #pragma unroll
    for(int i=thId;i<CBLMUN*2;i+=blockDim.x){
        if(i<(CBLMUN)){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow,i,K/LoNum) * GETELEMENT21(B_normmap,i,myBlockCol,N/LoNum);
            sC_bitmap[i] = norm_mul>Norm? 1:0; 
        }
        else{
            sC_bitmap[i] = 0;
        }
    }
    __syncthreads();

    for(int i=thId;i<CBLMUN;i+=blockDim.x){
        if(sC_bitmap[i]==1){
            int t=0;
            for(int j=0;j<i;j++){
                if(sC_bitmap[j]==1){
                    t++;
                }
            }
            sC_offset[t]=i;
        }
    }
    __syncthreads();

    for (unsigned int s = REDUCECBL/2; s > 0; s >>= 1) {
		if (thId < s) {
			sC_bitmap[thId] += sC_bitmap[thId + s];
		}
		__syncthreads();
    }
    const int validNum = sC_bitmap[0]; 

   
    int this_b,next_b;
    if(validNum>0){
        this_b=sC_offset[0];
        const int tempid=thId-16*32*(warpId/16);
        const int tempi=tempid/16;
        const int tempj=tempid%16*2;
        const float* matrix;
        float *smatrix;
        if(first16){
            smatrix=sA0;
            matrix=&GETELEMENT21(A,myBlockRowOff+tempi,this_b*LoNum+tempj,K);
        }
        else{
            smatrix=sB0;
            matrix=&GETELEMENT21(B,this_b*LoNum+tempi,myBlockColOff+tempj,K);
        }
        smatrix[tempid*2]=*(matrix);
        smatrix[tempid*2+1]=*(matrix+1);
    }
    else{
        return;
    }

    float * A_this_read=sA0;
    float * B_this_read=sB0;
    float * A_this_write=sA1;
    float * B_this_write=sB1;
    const int tempid=thId-32*16*(warpId/16)-32*8*(warpId/24);
    const int tempi=tempid/8;
    const int tempj=tempid%8*4;
    const float* matrix;
    float *smatrix;
    int ri=thId/16;
    int rj=thId%16*2;
    float myCresult1=0.0f,myCresult2=0.0f;

    #pragma unroll 
    for(int i=0;i<validNum;i++){
        __syncthreads(); 
        this_b = sC_offset[i];

        if(first16){

            float* mysA = &GETELEMENT21(A_this_read,ri,0,LoNum);
            float* mysB1 = &GETELEMENT21(B_this_read,0,rj,LoNum);
            float* mysB2 = &GETELEMENT21(B_this_read,0,rj+1,LoNum);
            
            #pragma unroll
            for(int i=0;i<LoNum;i++){ 
                myCresult1 += *(mysA+i) * *(mysB1+i*LoNum); 
                myCresult2 += *(mysA+i) * *(mysB2+i*LoNum); 
            }
        }
        else{
            if(i<validNum-1){
                next_b = sC_offset[i+1];
                const float* matrix;
                float *smatrix;
                if(warpId<24){
                    smatrix=&A_this_write[tempid*4];
                    matrix=&GETELEMENT21(A,myBlockRowOff+tempi,next_b*LoNum+tempj,K);
                }
                else{
                    smatrix=&B_this_write[tempid*4];
                    matrix=&GETELEMENT21(B,next_b*LoNum+tempi,myBlockColOff+tempj,K);
                }
                *(smatrix)=*(matrix);
                *(smatrix+1)=*(matrix+1);
                *(smatrix+2)=*(matrix+2);
                *(smatrix+3)=*(matrix+3);
            }
        }
        
        if(i%2==0){
            A_this_read=sA1;
            B_this_read=sB1;
            A_this_write=sA0;
            B_this_write=sB0;
        }
        else{
            A_this_read=sA0;
            B_this_read=sB0;
            A_this_write=sA1;
            B_this_write=sB1;
        }
    }

    if(first16){
        float* add=&GETELEMENT21(C,myBlockRowOff+ri,myBlockColOff+rj,N);
        *(add)=myCresult1;
        *(add+1)=myCresult2;
    }
    

}

__global__ void get_C_FP16_B32(const half* __restrict__ A,const float* __restrict__ A_normmap,const half* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset,float Norm){
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int kId = blockIdx.x;
    const int thId = threadIdx.x;
    const int warpId = thId/32;
    int REDUCECBL = 1<<(int)(log2(CBLMUN*1.0)+1);

    __shared__ int sC_bitmap[(CBLMUN/4+1)*4*2]; 
    __shared__ int sC_offset[(CBLMUN/4+1)*4];
    __shared__ half st[LoNum*LoNum];
    __shared__ half sA0[LoNum*LoNum],sB0[LoNum*LoNum]; 
    __shared__ half sA1[LoNum*LoNum],sB1[LoNum*LoNum]; 

    float norm_mul,myCresult=0.0f;
    const int myBlockRow = kId / (N/LoNum) + main_row_offset; 
    const int myBlockCol = kId % (N/LoNum); 
    const int myBlockRowOff = myBlockRow*LoNum;
    const int myBlockColOff = myBlockCol*LoNum;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a0_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b0_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a1_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b1_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    #pragma unroll
    for(int i=thId;i<REDUCECBL;i+=blockDim.x){
        if(i<(CBLMUN)){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow,i,K/LoNum) * GETELEMENT21(B_normmap,i,myBlockCol,N/LoNum);
            sC_bitmap[i] = norm_mul>=Norm? 1:0; 
        }
        else{
            sC_bitmap[i] = 0;
        }
    }
    __syncthreads();

    // reduce
    for(int i=thId;i<CBLMUN;i+=blockDim.x){
        if(sC_bitmap[i]==1){
            int t=0;
            for(int j=0;j<i;j++){
                if(sC_bitmap[j]==1){
                    t++;
                }
            }
            sC_offset[t]=i;
        }
    }
    __syncthreads();
    
    for (unsigned int s = REDUCECBL / 2; s > 0; s >>= 1) {
		if (thId < s) {
			sC_bitmap[thId] += sC_bitmap[thId + s];
		}
		__syncthreads();
    }
    const int validNum = sC_bitmap[0]; 
    
    const int warpi=warpId/2;
    const int warpj=warpId%2;
    const int myFinalRow16 = myBlockRow*2+warpi;
    const int myFinalCol16 = myBlockCol*2+warpj;

    int this_b,next_b;
    half * A_this_read=sA0;
    half * B_this_read=sB0;
    half * A_this_write=sA1;
    half * B_this_write=sB1;
    const int inWarpi = thId % 32 / 4 + warpId*8;
    const int inWarpj = (thId % 32 % 4)*8;
    if(validNum>0){
        this_b=sC_offset[0];
        #pragma unroll
        for(int line=warpId*8;line<(warpId+1)*8;line++){
            GETELEMENT21(A_this_read,line,thId%32,LoNum) = GETELEMENT21(A,myBlockRowOff+line,this_b*LoNum+thId%32,K);
            GETELEMENT21(B_this_read,line,thId%32,LoNum) = GETELEMENT21(B,this_b*LoNum+line,myBlockColOff+thId%32,N);
        }
    }

    #pragma unroll 
    for(int i=0;i<validNum;i++){
        __syncthreads();
        this_b = sC_offset[i];
        if(i+1<validNum){
            next_b = sC_offset[i+1];
            #pragma unroll
            for(int line=warpId*8;line<(warpId+1)*8;line++){
                GETELEMENT21(A_this_write,line,thId%32,LoNum) = GETELEMENT21(A,myBlockRowOff+line,next_b*LoNum+thId%32,K);
                GETELEMENT21(B_this_write,line,thId%32,LoNum) = GETELEMENT21(B,next_b*LoNum+line,myBlockColOff+thId%32,N);
            }
        }
        
        wmma::load_matrix_sync(a0_frag, GETOFF21(A_this_read,warpi*16,0*16,LoNum), LoNum);
        wmma::load_matrix_sync(b0_frag, GETOFF21(B_this_read,0*16,warpj*16,LoNum), LoNum);
        wmma::load_matrix_sync(a1_frag, GETOFF21(A_this_read,warpi*16,1*16,LoNum), LoNum);
        wmma::load_matrix_sync(b1_frag, GETOFF21(B_this_read,1*16,warpj*16,LoNum), LoNum);
        wmma::mma_sync(c_frag, a0_frag, b0_frag, c_frag);
        wmma::mma_sync(c_frag, a1_frag, b1_frag, c_frag);

        if(i%2==0){
            A_this_read=sA1;
            B_this_read=sB1;
            A_this_write=sA0;
            B_this_write=sB0;
        }
        else{
            A_this_read=sA0;
            B_this_read=sB0;
            A_this_write=sA1;
            B_this_write=sB1;
        }
    }

    wmma::store_matrix_sync(GETOFF21(C,myFinalRow16*16,myFinalCol16*16,N), c_frag, N,wmma::mem_row_major);
}


int main(int argc, char **argv){
    printf("input parameters: M=%d K=%d N=%d Norm=%f USINGHALF=%d\n",M,K,N,NormINIT,USINGHALF);

    int device_row_offset=K/LoNum/DEVICEDIM;
    if(K/LoNum/DEVICEDIM/PART<=0){
        printf("PART error! too many parts!\n");
        return;
    }

    TGPUplan      plan[DEVICEDIM];
    for(int i=0;i<DEVICEDIM;i++){
        cudaSetDevice(i);
        cudaStreamCreate(&plan[i].stream);
    }

    mytype *h_A = (mytype *)malloc(sizeof(mytype)*M*K);
    mytype *h_B = (mytype *)malloc(sizeof(mytype)*K*N);
    
    if(CNN||DECAY) {
        #if CNN
        getMatrixFromCSV(h_A,M,K,FILENAMEA);
        getMatrixFromCSV(h_B,K,N,FILENAMEB);
        #endif
        #if DECAY
        getMatrixFromMTX(h_A,M,K,FILENAMEA);
        getMatrixFromMTX(h_B,K,N,FILENAMEB);
        #endif
    }
    if(MATRIXNOR) getNormMatrix(h_A,h_B);
    if(MATRIXEXP){
        getDecayMatrixExp(h_A,1,0.1,M,K);
        getDecayMatrixExp(h_B,1,0.1,K,N);
    }
    if(MATRIXALG){
        getDecayMatrixAlg(h_A,0.1,0.1,M,K);
        getDecayMatrixAlg(h_B,0.1,0.1,K,N);
    }
    
    for(int i=0;i<DEVICEDIM;i++){
        cudaSetDevice(i);
        cudaMallocManaged((void **)&plan[i].h_A, sizeof(mytype)*M*K);
        cudaMallocManaged((void **)&plan[i].h_B, sizeof(mytype)*K*N);
        cudaMallocManaged((void **)&plan[i].h_C, sizeof(float)*M*N);
        cudaMallocManaged((void **)&plan[i].A_normmap, sizeof(float)*(M/LoNum)*(K/LoNum));
        cudaMallocManaged((void **)&plan[i].B_normmap, sizeof(float)*(K/LoNum)*(N/LoNum));

        cudaMemPrefetchAsync(plan[i].h_A, sizeof(mytype)*M*K, i); 
        cudaMemPrefetchAsync(plan[i].h_B, sizeof(mytype)*K*N, i);
        cudaMemPrefetchAsync(plan[i].h_C, sizeof(float)*M*N, i);
        cudaMemAdvise(plan[i].h_A, sizeof(mytype)*M*K, cudaMemAdviseSetReadMostly, i);
        cudaMemAdvise(plan[i].h_B, sizeof(mytype)*K*N, cudaMemAdviseSetReadMostly, i);

        cudaStreamCreate(&plan[i].stream);

        cudaMemcpy(plan[i].h_A,h_A,sizeof(mytype)*M*K,cudaMemcpyHostToDevice);
        cudaMemcpy(plan[i].h_B,h_B,sizeof(mytype)*K*N,cudaMemcpyHostToDevice);
    }
    
    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;

    #if SpAMM
    for(int i=0;i<TESTTIME;i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        

        const int partBlockOffset=M/LoNum/PART;
        int C_blocks = M*N/(LoNum*LoNum),C_threads=LoNum*LoNum;
        int A_blocks = M*K/(LoNum*LoNum),B_blocks = (K*N)/(LoNum*LoNum),F_threads = LoNum*LoNum;


        #pragma omp parallel num_threads(DEVICEDIM)
        { 
            int device = omp_get_thread_num();
            cudaSetDevice(device);

            for(int p=0;p<PART;p++){
                #if !USINGHALF
                if(LoNum==32){
                    unroll_get_Fnorm_pri<<<B_blocks/PART,F_threads/8,0,plan[device].stream>>>(plan[device].h_B,plan[device].B_normmap,K,N,p*partBlockOffset);
                }
                else{

                }
                
                #else
                if(LoNum==32){
                    unroll_get_Fnorm_FP16<<<B_blocks/PART,32*4,0,plan[device].stream>>>(plan[device].h_B,plan[device].B_normmap,K,N,p*partBlockOffset);
                }
                else{

                }
                #endif
                
                #if !USINGHALF
                if(LoNum==32){
                    unroll_get_Fnorm_pri<<<A_blocks/DEVICEDIM/PART,F_threads/8,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,M,K,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                }
                else{

                }
                
                #else
                if(LoNum==32){
                    unroll_get_Fnorm_FP16<<<A_blocks/DEVICEDIM/PART,32*4,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,M,K,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                }
                else{

                }
                #endif
            }

            cudaStreamSynchronize(plan[device].stream);

            #if TUNINGFLAG
            Norm = tuneValidRate(plan[device].A_normmap,plan[device].B_normmap,M/DEVICEDIM/PART,N);
            #else
            Norm = NormINIT;
            #endif

            for(int p=0;p<PART;p++){
                #if !USINGHALF
                if(LoNum==32){
                    get_C_Threads1Element_Mul<<<C_blocks/DEVICEDIM/PART,C_threads,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,plan[device].h_B,plan[device].B_normmap,plan[device].h_C,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM),Norm);
                }
                else{

                }
                
                #else
                if(LoNum==32){
                    get_C_FP16_B32<<<C_blocks/DEVICEDIM/PART,32*4,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,plan[device].h_B,plan[device].B_normmap,plan[device].h_C,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM),Norm);
                }
                else{

                }
                #endif
            }
        }

        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i>WARMUP) sum += elapsed; 
    }

    
    double spammtime=sum/((TESTTIME-WARMUP));
    printf("SpAMM time=%fs\n",spammtime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif
    
    
}
