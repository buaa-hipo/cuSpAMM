
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

__global__ void unroll_get_Fnorm_pri_FP16(const half* __restrict__ A,float *A_normmap,int m,int n,int blockRowOff){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ float sdata[LoNum*LoNum];

    int valid;
    const int myBlockRow = kId / (CBLMUN)+blockRowOff;
    const int myBlockCol = kId % (CBLMUN);
    const int myBlockId = myBlockRow*(n/LoNum)+myBlockCol;
    const int myThreadRow = thId / 32;
    const int myThreadCol = thId % 32;
    const int myFinalRow = myBlockRow*LoNum+myThreadRow;
    const int myFinalCol = myBlockCol*LoNum+myThreadCol;
    const int reduceNum = blockDim.x;

    //每个线程取8个
    valid = id > m*n? 0:1;
    if(valid){
        int tadd = myFinalRow*n+myFinalCol;
        float t1 = __half2float(A[tadd]);
        sdata[thId] = t1*t1;
        // if(thId==0&&id==0){
        //     printf("%f %f %f %f %f %f %f %f\n",t1,t2,t3,t4,t5,t5,t6,t7,t8);
        //     printf("%f\n",__half2float(A[tadd]));
        // } 
    } 
    __syncthreads();

    for (unsigned int s = reduceNum / 2; s >= 1; s >>= 1) {
		if (thId < s) {
			sdata[thId] += sdata[thId + s];
		}
		__syncthreads();
    }
    
    if(thId==0){
        A_normmap[myBlockId] = sqrt(sdata[0]);
        // printf("shared mem kid=%d val=%f %f\n",kId,sqrt(sdata[0]),A_normmap[myBlockId]);
    } 
}

__global__ void unroll_get_Fnorm_pri(mytype* A,float *A_normmap,int m,int n,int blockRowOff){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ float sdata[LoNum*LoNum];

    int valid;
    const int myBlockRow = kId / (CBLMUN)+blockRowOff;
    const int myBlockCol = kId % (CBLMUN);
    const int myBlockId = myBlockRow*(n/LoNum)+myBlockCol;
    const int myThreadRow = thId / 32;
    const int myThreadCol = thId % 32;
    const int myFinalRow = myBlockRow*LoNum+myThreadRow;
    const int myFinalCol = myBlockCol*LoNum+myThreadCol;
    const int reduceNum = blockDim.x;

    //每个线程取8个
    valid = id > m*n? 0:1;
    if(valid){
        int tadd = myFinalRow*n+myFinalCol;
        float t1 = A[tadd];
        sdata[thId] = t1*t1;
        // if(thId==0&&id==0){
        //     printf("%f %f %f %f %f %f %f %f\n",t1,t2,t3,t4,t5,t5,t6,t7,t8);
        //     printf("%f\n",__half2float(A[tadd]));
        // } 
    } 
    __syncthreads();

    for (unsigned int s = reduceNum / 2; s >= 1; s >>= 1) {
		if (thId < s) {
			sdata[thId] += sdata[thId + s];
		}
		__syncthreads();
    }
    
    if(thId==0){
        A_normmap[myBlockId] = sqrt(sdata[0]);
        // printf("shared mem kid=%d val=%f %f\n",kId,sqrt(sdata[0]),A_normmap[myBlockId]);
    } 
}

__global__ void get_C_Threads1Element_Mul(float* A,float* A_normmap,float*  B,float* B_normmap,float* C,const int main_row_offset,float Norm){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ int sC_bitmap[CBLMUN];//share mem需要初始化！！

    float norm_mul,myCresult=0.0f;
    const int myBlockRow = kId / (CBLMUN) + main_row_offset; 
    const int myBlockCol = kId % (CBLMUN); //负责计算块坐标C[Brow,Bcol]处的块
    const int myBlockRowOff = myBlockRow*LoNum;
    const int myBlockColOff = myBlockCol*LoNum;
    const int myThreadRow = thId/LoNum;
    const int myThreadCol = thId%LoNum;
    const int myFinalRow = myBlockRowOff+myThreadRow;
    const int myFinalCol = myBlockColOff+myThreadCol;
    

    //遍历bitmap,每个线程负责一个位置的元素
    #pragma unroll 
    for(int b=0;b<CBLMUN;b++){
        if(thId==0){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow,b,CBLMUN) * GETELEMENT21(B_normmap,b,myBlockCol,CBLMUN);
            sC_bitmap[b] = norm_mul>Norm? 1:0;
        }
        __syncthreads();
        if(sC_bitmap[b]==1){//慢

            #pragma unroll 
            for(int i=0;i<LoNum;i++){ //极慢，三倍
                myCresult +=GETELEMENT21(A,myFinalRow,b*LoNum+i,K)*GETELEMENT21(B,b*LoNum+i,myFinalCol,N);
            }
        }
    }

    GETELEMENT21(C,myFinalRow,myFinalCol,N) = myCresult; 

}

//每个kernel计算C[LoNum,LoNum]
//静态无分配版本，每个线程一个元素进行计算，LoNum*LoNum个线程
__global__ void get_C_Threads1Element_Mul_FP16(const half* __restrict__ A,const float* __restrict__ A_normmap,const half* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset,float Norm){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ int sC_bitmap[CBLMUN];//share mem需要初始化！！

    float norm_mul;
    half myCresult=__half2float(0.0f);
    const int myBlockRow = kId / (CBLMUN) + main_row_offset; 
    const int myBlockCol = kId % (CBLMUN); //负责计算块坐标C[Brow,Bcol]处的块
    const int myBlockRowOff = myBlockRow*LoNum;
    const int myBlockColOff = myBlockCol*LoNum;
    const int myThreadRow = thId/LoNum;
    const int myThreadCol = thId%LoNum;
    const int myFinalRow = myBlockRowOff+myThreadRow;
    const int myFinalCol = myBlockColOff+myThreadCol;
    

    //遍历bitmap,每个线程负责一个位置的元素
    #pragma unroll 
    for(int b=0;b<CBLMUN;b++){
        if(thId==0){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow,b,CBLMUN) * GETELEMENT21(B_normmap,b,myBlockCol,CBLMUN);
            sC_bitmap[b] = norm_mul>Norm? 1:0;
        }
        __syncthreads();
        if(sC_bitmap[b]==1){//慢

            #pragma unroll 
            for(int i=0;i<LoNum;i++){ //极慢，三倍
                myCresult +=GETELEMENT21(A,myFinalRow,b*LoNum+i,K)*GETELEMENT21(B,b*LoNum+i,myFinalCol,N);
            }
        }
    }

    GETELEMENT21(C,myFinalRow,myFinalCol,N) = myCresult; 
}



int main(int argc, char **argv){

    int device_row_offset=K/LoNum/DEVICEDIM;
    //测试part是否太大
    if(K/LoNum/DEVICEDIM/PART<=0){
        printf("PART error! too many parts!\n");
        return;
    }

    TGPUplan      plan[DEVICEDIM];
    for(int i=0;i<DEVICEDIM;i++){
        cudaSetDevice(i);
        cudaStreamCreate(&plan[i].stream);
    }

    //统一内存h_A,h_B;
    mytype *h_A = (mytype *)malloc(sizeof(mytype)*M*K);
    mytype *h_B = (mytype *)malloc(sizeof(mytype)*K*N);
    printf("%d %d %d\n",M,K,N);
    
    // //给A,B赋值
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
    // printf("---A---\n");MATRIXSHOW21D(h_A,M,K);

    for(int i=0;i<DEVICEDIM;i++){
        //给私有的bitmap和C分配空间，C用UM
        cudaSetDevice(i);
        cudaMallocManaged((void **)&plan[i].h_A, sizeof(mytype)*M*K);
        cudaMallocManaged((void **)&plan[i].h_B, sizeof(mytype)*K*N);
        cudaMallocManaged((void **)&plan[i].h_C, sizeof(float)*M*N);
        cudaMallocManaged((void **)&plan[i].A_normmap, sizeof(float)*(M/LoNum)*(K/LoNum));
        cudaMallocManaged((void **)&plan[i].B_normmap, sizeof(float)*(K/LoNum)*(N/LoNum));

        //流
        cudaStreamCreate(&plan[i].stream);

        //拷贝数据
        cudaMemcpy(plan[i].h_A,h_A,sizeof(mytype)*M*K,cudaMemcpyHostToDevice);
        cudaMemcpy(plan[i].h_B,h_B,sizeof(mytype)*K*N,cudaMemcpyHostToDevice);
    }

    printf("INIT DONE--------------\n");
    printf("para: M=%d K=%d N=%d Norm=%f DEVICE=%d PARTS=%d \nALG=%d EXP=%d CNN=%d DECAY=%d\n",M,K,N,NormINIT,DEVICEDIM,PART,MATRIXALG,MATRIXEXP,CNN,DECAY);
    printf("TUNINGFLAG=%d ExpectedRate=%f TUNINGTIME=%d TUNINGERROR=%f\n",TUNINGFLAG,ExpectedRate,TUNINGTIME,TUNINGERROR);
    //计时部分
    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;

    #if SpAMM
    for(int i=0;i<TESTTIME;i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        #pragma unroll 2
        for(int device=0;device<DEVICEDIM;device++){
            cudaSetDevice(device);

            const int partBlockOffset=M/LoNum/PART; //所有行分P次算
            //计算全部B范数
            int A_blocks = M*K/(LoNum*LoNum),B_blocks = (K*N)/(LoNum*LoNum),F_threads = LoNum*LoNum;
            for(int p=0;p<PART;p++){
                #if !USINGHALF
                if(LoNum==32){
                    unroll_get_Fnorm_pri<<<B_blocks/PART,32*32,0,plan[device].stream>>>(plan[device].h_B,plan[device].B_normmap,K,N,p*partBlockOffset);
                }
                else{

                }
                
                #else
                if(LoNum==32){
                    unroll_get_Fnorm_pri_FP16<<<B_blocks/PART,32*32,0,plan[device].stream>>>(plan[device].h_B,plan[device].B_normmap,K,N,p*partBlockOffset);
                }
                else{

                }
                #endif
            }
            cudaStreamSynchronize(plan[device].stream);
            // printf("---the normmap of B:---\n");
            // MATRIXSHOW21D(plan[device].B_normmap,B_blocks,1);


            //计算某几行A范数和C结果
            int C_blocks = M*N/(LoNum*LoNum),C_threads=LoNum*LoNum;
            for(int p=0;p<PART;p++){
                #if !USINGHALF
                if(LoNum==32){
                    unroll_get_Fnorm_pri<<<A_blocks/DEVICEDIM/PART,32*32,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,M,K,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                }
                else{

                }
                
                #else
                if(LoNum==32){
                    unroll_get_Fnorm_pri_FP16<<<A_blocks/DEVICEDIM/PART,32*32,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,M,K,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                }
                else{

                }
                #endif

                cudaStreamSynchronize(plan[device].stream);
                // printf("---the normmap of A:---\n");
                // MATRIXSHOW21D(plan[device].A_normmap,B_blocks,1);

                #if TUNINGFLAG
                Norm = tuneValidRate(plan[device].A_normmap,plan[device].B_normmap,M/DEVICEDIM/PART,N);
                #else
                Norm = NormINIT;
                #endif

                #if !USINGHALF
                if(LoNum==32){
                    get_C_Threads1Element_Mul<<<C_blocks/DEVICEDIM/PART,C_threads,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,plan[device].h_B,plan[device].B_normmap,plan[device].h_C,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM),Norm);
                }
                else{

                }
                
                #else
                if(LoNum==32){
                    get_C_Threads1Element_Mul_FP16<<<C_blocks/DEVICEDIM/PART,C_threads,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,plan[device].h_B,plan[device].B_normmap,plan[device].h_C,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM),Norm);
                }
                else{

                }
                #endif
            }
        }

        // //host同步
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i>=WARMUP) sum += elapsed; //测速的时候改成3
    }

    printf("spammm time=%fs\n",sum/(TESTTIME-WARMUP));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif
    
    #if CUBLAS
    run_cublas_time(h_A,h_B);
    #endif
    
    // //检验结果
    #if SpAMM
    if(CHECK) {
        //整合最终C的结果,C永远是float
        float* result_C;
        cudaMallocManaged((void **)&result_C, sizeof(float)*M*N);
        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                result_C[i*N+j]=plan[i/(M/DEVICEDIM)].h_C[i*N+j];
            }
        }
        // MATRIXSHOW21D(result_C,M,N);
        // printf("hahah %f\n",result_C[1024]);
        check_simple_gpu(h_A,h_B,result_C);
        

        //取0号的normmap验证
        float *h_Amap;
        cudaMallocManaged((void **)&h_Amap, sizeof(float)*M*K/LoNum/LoNum);
        const int ndim = M*K/LoNum/LoNum/DEVICEDIM;
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
    #endif

    // printf("---NORM squrt A:---\n"); MATRIXSHOW21D(A_normmap,CBLMUN,CBLMUN);
    // printf("---NORM squrt B:---\n"); MATRIXSHOW21D(B_normmap,CBLMUN,CBLMUN);
    // printf("!!! NORM mul setting = %f!!!\n\n",Norm);
    
    //end
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);
}
