
#include "main.h"

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

//使用原语reduce，每个warp自加，然后给前32个再累加
//每个kernel LoNum*LoNum/8个线程
__global__ void unroll_get_Fnorm_pri(const float* __restrict__ A,float *A_normmap,int m,int n,int blockRowOff){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ float sdata[LoNum*LoNum/8/32];

    int valid=0;
    const int myBlockRow = kId / (CBLMUN)+blockRowOff;
    const int myBlockCol = kId % (CBLMUN);
    const int myBlockId = myBlockRow*(T/LoNum)+myBlockCol;
    const int myThreadRow = thId / (LoNum/8);
    const int myThreadCol = thId % (LoNum/8);
    const int myFinalRow = myBlockRow*LoNum+myThreadRow;
    const int myFinalCol = myBlockCol*LoNum+myThreadCol*8;

    //每个线程取1个
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
        // if(thId%32==0) printf("thid=%d warpid=%d inwarpid=%d val=%f\n",thId,thId/32,thId%32,val);
    }       
    if(thId%32==0){
        sdata[thId/32]=val;
        // printf("%d %d val=%f dim=%d\n",thId,thId/32,sdata[thId/32],blockDim.x);
    } 
    
    __syncthreads();
    float r=0;
    
    if (thId < blockDim.x/32)
    {
        // printf("thid=%d val=%f sw[thid]=%f\n",thId,val,sdata[thId]);
        val=sdata[thId];
        // printf("%d %f\n",thId,sdata[thId]);
        for (int offset = blockDim.x/32/2; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    if(thId==0){
        A_normmap[myBlockId] = sqrt(val); //记得开方
        // printf("pri kid=%d val=%f\n",kId,val);
    } 
}

//4*32
__global__ void unroll_get_Fnorm_FP16(const half* __restrict__ A,float *A_normmap,int m,int n,int blockRowOff){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    int warpId = thId / 32;

    const int myBlockRow = kId / (CBLMUN)+blockRowOff;
    const int myBlockCol = kId % (CBLMUN);
    const int myBlockId = myBlockRow*(T/LoNum)+myBlockCol;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> chalf_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    __shared__ half sdata_half[32*32];
    __shared__ float sdata_float[4];
    
    //要算32*32的范数和,块坐标(myBlockRow,myBlockCol)，每个warp算16*16
    int warpi=warpId/2;
    int warpj=warpId%2;
    
    //setFragment(A, 1.0);
    // setFragment(C, 0.0);
    // loadMatrix(B, X+offset);//+矩阵元素平方操作
    // MMA(C, A, B, C);
    wmma::fill_fragment(a_frag, 1.0f);
    wmma::fill_fragment(chalf_frag, 0.0f);
    wmma::load_matrix_sync(b_frag, GETOFF21(A,myBlockRow*LoNum+warpi*16,myBlockCol*LoNum+warpj*16,K), K);
    for (int i = 0; i < b_frag.num_elements; i++) {
        half t=b_frag.x[i];
        b_frag.x[i] = __float2half(__half2float(t) * __half2float(t));
    }
    wmma::mma_sync(chalf_frag, a_frag, b_frag, chalf_frag);
    // copyFromTo(C, A);
    wmma::store_matrix_sync(GETOFF21(sdata_half,warpi*16,warpj*16,32), chalf_frag, 32,wmma::mem_row_major);
    __syncthreads();
    
    wmma::load_matrix_sync(a_frag, GETOFF21(sdata_half,warpi*16,warpj*16,32), 32);
    
    for (int i = 0; i < a_frag.num_elements; i++) {
        // if(warpId==3) printf("thid=%d %f\n",thId,__half2float(a_frag.x[i]));
    }
    // // setFrament(B, 1.0);
    // // setFrament(C, 0.0);
    // // MMA(C, A, B, C);
    wmma::fill_fragment(b_frag, 1.0f);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
 
    __syncthreads();
    if(thId%32==0){
        sdata_float[warpId]=c_frag.x[0];
    }
    __syncthreads();
    if(thId==0){
        // for(int i=0;i<4;i++){
        //     printf("kid=%d i=%d %f\n",kId,i,sdata_float[i]);
        // }
        A_normmap[myBlockId]=sqrt(sdata_float[0]+sdata_float[1]+sdata_float[2]+sdata_float[3]);
        // printf("%d %f\n",kId,A_normmap[myBlockId]);
    }
}

//每个kernel计算C[LoNum,LoNum]
//静态无分配版本，每个线程一个元素进行计算
__global__ void get_C_Threads1Element_Mul(const float* __restrict__ A,const float* __restrict__ A_normmap,const float* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset){
    printf("haha\n");
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ int sC_bitmap[CBLMUN];//share mem需要初始化！！
    __shared__ float sA[LoNum*LoNum],sB[LoNum*LoNum]; //sC可以换成局部变量，但有local的风险
    sA[thId]=1;
    // float norm_mul,myCresult=0.0f;
    // const int myBlockRow = kId / (CBLMUN) + main_row_offset; 
    // const int myBlockCol = kId % (CBLMUN); //负责计算块坐标C[Brow,Bcol]处的块
    // const int myBlockRowOff = myBlockRow*LoNum;
    // const int myBlockColOff = myBlockCol*LoNum;
    // const int myThreadRow = thId/LoNum;
    // const int myThreadCol = thId%LoNum;
    // const int myFinalRow = myBlockRowOff+myThreadRow;
    // const int myFinalCol = myBlockColOff+myThreadCol;
    
    
    // //需要A_norm第R行，B_norm第C列
    // #pragma unroll
    // for(int i=thId;i<CBLMUN;i+=blockDim.x){
    //     if(thId<(CBLMUN)){
    //         norm_mul = GETELEMENT21(A_normmap,myBlockRow,i,CBLMUN) * GETELEMENT21(B_normmap,i,myBlockCol,CBLMUN);
    //         sC_bitmap[i] = norm_mul>Norm? 1:0; //!范数计算有E的浮动误差，应该是6位有效数字
    //     }
    // }
    // __syncthreads();//不能和下面合并！因为有的线程的b可能没算完就结束了，但是非常费时间

    // //遍历bitmap,每个线程负责一个位置的元素
    // #pragma unroll 
    // for(int b=0;b<CBLMUN;b++){
    //     if(sC_bitmap[b]==1){//慢
    //         __syncthreads(); //等待算完，不然会有线程改变s值
    //         //[计算32*32规模的矩阵乘]
    //         //共同加载share A(mBR)行第b个块,B(mBC)列第b个块
    //         sA[thId] = GETELEMENT21(A,myFinalRow,b*LoNum+myThreadCol,K);//慢
    //         sB[thId] = GETELEMENT21(B,b*LoNum+myThreadRow,myFinalCol,N);

    //         __syncthreads();
            
    //         //矩阵小块(LoNum,LoNum)乘 每个线程算C内[thId/L,thId%L]处的最后值
    //         float* mysA = &GETELEMENT21(sA,myThreadRow,0,LoNum);//sA第myTR行，sB第myTC列
    //         float* mysB = &GETELEMENT21(sB,0,myThreadCol,LoNum);

    //         #pragma unroll 
    //         for(int i=0;i<LoNum;i++){ //极慢，三倍
    //             myCresult += *(mysA+i) * *(mysB+i*LoNum); 
    //         }
    //     }
    // }

    // GETELEMENT21(C,myFinalRow,myFinalCol,N) = myCresult; 
    // if(kId==0&&thId==0) printf("kid=%d thid=%d %f\n",kId,thId,myCresult);
}

//4个warp，计算32*32
__global__ void get_C_FP16_B32(const half* __restrict__ A,const float* __restrict__ A_normmap,const half* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;
    int thId = threadIdx.x;
    int warpId = thId/32;
    __shared__ int sC_bitmap[CBLMUN];//share mem需要初始化！！
    // __shared__ half sA[LoNum*LoNum],sB[LoNum*LoNum]; //sC可以换成局部变量，但有local的风险

    float norm_mul,myCresult=0.0f;
    const int myBlockRow = kId / (CBLMUN) + main_row_offset; 
    const int myBlockCol = kId % (CBLMUN); //负责计算块坐标C[Brow,Bcol]处的块
    const int myBlockRowOff = myBlockRow*LoNum;
    const int myBlockColOff = myBlockCol*LoNum;
    
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    //需要A_norm第R行，B_norm第C列
    #pragma unroll
    for(int i=thId;i<CBLMUN;i+=blockDim.x){
        if(thId<(CBLMUN)){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow,i,CBLMUN) * GETELEMENT21(B_normmap,i,myBlockCol,CBLMUN);
            sC_bitmap[i] = norm_mul>Norm? 1:0; //!范数计算有E的浮动误差，应该是6位有效数字
        }
    }
    __syncthreads();//不能和下面合并！因为有的线程的b可能没算完就结束了，但是非常费时间

    int warpi=warpId/2;
    int warpj=warpId%2;
    const int myFinalRow16 = myBlockRow*2+warpi;
    const int myFinalCol16 = myBlockCol*2+warpj;
    //遍历bitmap,每个线程负责一个位置的元素
    #pragma unroll 
    for(int b=0;b<CBLMUN;b++){
        if(sC_bitmap[b]==1){//慢
            
            //以warp为单位，每个计算最终16*16，算两个
            //计算大块为C[myBlockRow][myBlockCol][warpi][warpj]+=A[mR][b][warpi][k]*B[b][mC][b][warpj];
            for(int k=0;k<2;k++){
                wmma::load_matrix_sync(a_frag, GETOFF21(A,myFinalRow16*16,(b*LoNum/16+k)*16,K), K);
                // wmma::fill_fragment(a_frag, 1.0f);
                wmma::load_matrix_sync(b_frag, GETOFF21(B,(b*LoNum/16+k)*16,myFinalCol16*16,K), K);
                // if(warpId==0&&thId%32==0) printf("warpi=%d warpj=%d b=%d k=%d addB=(%d,%d) b[0]=%f\n",warpi,warpj,b,k,(b*LoNum/16+k)*16,myFinalCol16*16,__half2float(GETELEMENT21(B,(b*LoNum/16+k)*16,myFinalCol16*16+1,16)));
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
    }
    wmma::store_matrix_sync(GETOFF21(C,myFinalRow16*16,myFinalCol16*16,K), c_frag, K,wmma::mem_row_major);

}

int main(int argc, char **argv){

    int device_row_offset=T/LoNum/DEVICEDIM;
    //测试part是否太大
    if(T/LoNum/DEVICEDIM/PART<=0){
        printf("PART error! too many parts!\n");
        return;
    }
    if(LoNum%32!=0||T/LoNum<0||T%LoNum!=0){
        printf("Dim error! invalid LoNum or T\n");
    }

    TGPUplan      plan[DEVICEDIM];
    for(int i=0;i<DEVICEDIM;i++){
        cudaSetDevice(i);
        cudaStreamCreate(&plan[i].stream);
    }

    //统一内存h_A,h_B;
    mytype *h_A = (mytype *)malloc(sizeof(mytype)*T*T);
    mytype *h_B = (mytype *)malloc(sizeof(mytype)*T*T);
    
    //给A,B赋值
    if(MATRIXNOR) getNormMatrix(h_A,h_B);
    if(MATRIXEXP){
        getDecayMatrixExp(h_A,1,0.1,M,K);
        getDecayMatrixExp(h_B,1,0.1,K,N);
    }
    if(MATRIXALG){
        getDecayMatrixAlg(h_A,0.1,0.1,K,N);
        getDecayMatrixAlg(h_B,0.1,0.1,K,N);
    }
    // printf("---A---\n");MATRIXSHOW21D(h_A,M,K);

    for(int i=0;i<DEVICEDIM;i++){
        //给私有的bitmap和C分配空间，C用UM
        cudaSetDevice(i);
        cudaMallocManaged((void **)&plan[i].h_A, sizeof(mytype)*T*T);
        cudaMallocManaged((void **)&plan[i].h_B, sizeof(mytype)*T*T);
        cudaMallocManaged((void **)&plan[i].h_C, sizeof(float)*T*T);
        cudaMallocManaged((void **)&plan[i].A_normmap, sizeof(float)*(T/LoNum)*(T/LoNum));
        cudaMallocManaged((void **)&plan[i].B_normmap, sizeof(float)*(T/LoNum)*(T/LoNum));

        //UM指导
        cudaMemPrefetchAsync(plan[i].h_A, sizeof(mytype)*T*T, i);
        cudaMemPrefetchAsync(plan[i].h_B, sizeof(mytype)*T*T, i);
        cudaMemPrefetchAsync(plan[i].h_C, sizeof(float)*T*T, i);
        cudaMemAdvise(plan[i].h_A, sizeof(mytype)*T*T, cudaMemAdviseSetReadMostly, i);
        cudaMemAdvise(plan[i].h_B, sizeof(mytype)*T*T, cudaMemAdviseSetReadMostly, i);

        //流
        cudaStreamCreate(&plan[i].stream);

        //拷贝数据
        cudaMemcpy(plan[i].h_A,h_A,sizeof(mytype)*T*T,cudaMemcpyHostToDevice);
        cudaMemcpy(plan[i].h_B,h_B,sizeof(mytype)*T*T,cudaMemcpyHostToDevice);
    }

    printf("INIT DONE--------------\n");
    printf("para: T=%d Norm=%f DEVICE=%d PARTS=%d ALG=%d EXP=%d\n",T,Norm,DEVICEDIM,PART,MATRIXALG,MATRIXEXP);
    //计时部分
    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;

    for(int i=0;i<TESTTIME;i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        #pragma unroll 2
        for(int device=0;device<DEVICEDIM;device++){
            cudaSetDevice(device);

            const int partBlockOffset=T/LoNum/PART; //所有行分P次算
            //计算全部B范数
            int A_blocks = M*K/(LoNum*LoNum),B_blocks = (K*N)/(LoNum*LoNum),F_threads = LoNum*LoNum;
            for(int p=0;p<PART;p++){
                #if !USINGHALF
                unroll_get_Fnorm_pri<<<B_blocks/PART,F_threads/8,0,plan[device].stream>>>(plan[device].h_B,plan[device].B_normmap,M,K,p*partBlockOffset);
                // if(p!=PART-1) cudaStreamSynchronize(plan[device].stream);
                #else
                unroll_get_Fnorm_FP16<<<B_blocks/PART,32*4,0,plan[device].stream>>>(plan[device].h_B,plan[device].B_normmap,M,K,p*partBlockOffset);
                #endif
            }
            cudaStreamSynchronize(plan[device].stream);
            // printf("---the normmap of B:---\n");
            // MATRIXSHOW21D(plan[device].B_normmap,B_blocks,1);


            //计算某几行A范数和C结果
            int C_blocks = M*N/(LoNum*LoNum),C_threads=LoNum*LoNum;
            for(int p=0;p<PART;p++){
                #if !USINGHALF
                unroll_get_Fnorm_pri<<<A_blocks/DEVICEDIM/PART,F_threads/8,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,M,K,device*(T/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                #else
                unroll_get_Fnorm_FP16<<<A_blocks/DEVICEDIM/PART,32*4,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,M,K,device*(T/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                #endif

                cudaStreamSynchronize(plan[device].stream);

                #if !USINGHALF
                get_C_Threads1Element_Mul<<<C_blocks/DEVICEDIM/PART,32,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,plan[device].h_B,plan[device].B_normmap,plan[device].h_C,device*(T/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                #else
                get_C_FP16_B32<<<C_blocks/DEVICEDIM/PART,32*4,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,plan[device].h_B,plan[device].B_normmap,plan[device].h_C,device*(T/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                #endif
            }
        }

        // //host同步
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i!=0) sum += elapsed;
    }
    
    printf("time=%fs\n",sum/(TESTTIME-1));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
    // //检验结果
    if(CHECK) {
        //整合最终C的结果,C永远是float
        float* result_C;
        cudaMallocManaged((void **)&result_C, sizeof(float)*T*T);
        for(int i=0;i<T;i++){
            for(int j=0;j<T;j++){
                result_C[i*T+j]=plan[i/(T/DEVICEDIM)].h_C[i*T+j];
            }
        }
        // MATRIXSHOW21D(result_C,M,N);
        check_simple_gpu(h_A,h_B,result_C);
        


        //取0号的normmap验证
        float *h_Amap;
        cudaMallocManaged((void **)&h_Amap, sizeof(float)*T*T/LoNum/LoNum);
        const int ndim = T*T/LoNum/LoNum/DEVICEDIM;
        for(int device=0;device<DEVICEDIM;device++){
            for(int i=0;i<ndim;i++){
                h_Amap[i+device*ndim] = plan[device].A_normmap[i+device*ndim];
            }
        }
        countValid(h_Amap,plan[0].B_normmap);
        // printf("A norm");
        // checkNormMap(h_A,h_Amap);//测试范数
        // printf("B norm");
        // checkNormMap(h_B,h_Bmap);//测试范数
        
    }

    // printf("---NORM squrt A:---\n"); MATRIXSHOW21D(A_normmap,CBLMUN,CBLMUN);
    // printf("---NORM squrt B:---\n"); MATRIXSHOW21D(B_normmap,CBLMUN,CBLMUN);
    // printf("!!! NORM mul setting = %f!!!\n\n",Norm);
    
    //end
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);
}
