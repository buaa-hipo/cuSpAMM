
#include "main.h"


//Mutiple-GPU Plan Structure
typedef struct
{
    float *h_A, *h_B;
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



#define SHAREFLAG 1
//输入：矩阵A，规模m，n(不能写宏定义因为要复用)，范数锁norm
//每个kernel算[LoNum * LoNum]大小的矩阵范数,每个线程取得一个元素然后reduce
__global__ void get_Fnorm(float *A,float *A_normmap,int m,int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ float sdata[LoNum*LoNum];

    int valid;
    float t;
    int myBlockRow = kId / (CBLMUN);
    int myBlockCol = kId % (CBLMUN);
    int myThreadRow = thId / LoNum;
    int myThreadCol = thId % LoNum;

    //每个线程取一个【待优化，可以取多个】
    valid = id > m*n? 0:1;
    if(valid){
        t = GETELEMENT21(A,myBlockRow*LoNum+myThreadRow,myBlockCol*LoNum+myThreadCol,n);
        sdata[thId] = t*t;
    } 
    __syncthreads();

    //naive版reduce，算完完整的范数
    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (thId < s)
        {
            sdata[thId] += sdata[thId + s];
        }
         __syncthreads();        // make sure all adds at one stage are done!
    }

    if(thId==0) A_normmap[kId] = sqrt(sdata[0]);
}

//每个线程算8个元素,同行相邻的! 每个kernel算一个[LoNum][LoNum]的范数
//每个kernel LoNum*LoNum/8个线程
//!虽然有bank conflict，但是是最快的
__global__ void unroll_get_Fnorm(const float* __restrict__ A,float *A_normmap,int m,int n,int blockRowOff){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ float sdata[LoNum*LoNum];

    int valid;
    const int myBlockRow = kId / (CBLMUN)+blockRowOff;
    const int myBlockCol = kId % (CBLMUN);
    const int myBlockId = myBlockRow*(T/LoNum)+myBlockCol;
    const int myThreadRow = thId*8 / LoNum;
    const int myThreadCol = thId*8 % LoNum;
    const int myFinalRow = myBlockRow*LoNum+myThreadRow;
    const int myFinalCol = myBlockCol*LoNum+myThreadCol;
    const int reduceNum = blockDim.x;

    //每个线程取8个
    valid = id*8 > m*n? 0:1;
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
        sdata[thId] = t1*t1+t2*t2+t3*t3+t4*t4+t5*t5+t6*t6+t7*t7+t8*t8;
    } 
    __syncthreads();

    for (unsigned int s = reduceNum / 2; s > 32; s >>= 1) {
		if (thId < s) {
			sdata[thId] += sdata[thId + s];
		}
		__syncthreads();
    }
    
    if (thId < 32)
    {
        volatile float* sw = sdata;
        sw[thId] += sw[thId + 32];
        sw[thId] += sw[thId + 16];
        sw[thId] += sw[thId + 8];
        sw[thId] += sw[thId + 4];
        sw[thId] += sw[thId + 2];
        sw[thId] += sw[thId + 1];
    }
    if(thId==0) A_normmap[myBlockId] = sqrt(sdata[0]);
}

//每个kernel计算C[LoNum,LoNum]
//静态无分配版本，每个线程一个元素进行计算
//B normmap !列索引 待优化! ; 分配乘法任务待优化
__global__ void get_C_Threads1Element_Mul(const float* __restrict__ A,const float* __restrict__ A_normmap,const float* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ int sC_bitmap[CBLMUN];//share mem需要初始化！！
    __shared__ float sA[LoNum*LoNum],sB[LoNum*LoNum]; //sC可以换成局部变量，但有local的风险

    float norm_mul,myCresult=0.0f;
    const int myBlockRow = kId / (CBLMUN) + main_row_offset; 
    const int myBlockCol = kId % (CBLMUN); //负责计算块坐标C[Brow,Bcol]处的块
    const int myBlockRowOff = myBlockRow*LoNum;
    const int myBlockColOff = myBlockCol*LoNum;
    const int myThreadRow = thId/LoNum;
    const int myThreadCol = thId%LoNum;
    const int myFinalRow = myBlockRowOff+myThreadRow;
    const int myFinalCol = myBlockColOff+myThreadCol;
    
    
    //需要A_norm第R行，B_norm第C列
    #pragma unroll
    for(int i=thId;i<CBLMUN;i+=blockDim.x){
        if(thId<(CBLMUN)){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow,i,CBLMUN) * GETELEMENT21(B_normmap,i,myBlockCol,CBLMUN);
            sC_bitmap[i] = norm_mul>Norm? 1:0; //!范数计算有E的浮动误差，应该是6位有效数字
        }
    }
    __syncthreads();//不能和下面合并！因为有的线程的b可能没算完就结束了，但是非常费时间

    //遍历bitmap,每个线程负责一个位置的元素
    #pragma unroll 
    for(int b=0;b<CBLMUN;b++){
        if(sC_bitmap[b]==1){//慢
            __syncthreads(); //等待算完，不然会有线程改变s值
            //[计算32*32规模的矩阵乘]
            //共同加载share A(mBR)行第b个块,B(mBC)列第b个块
            sA[thId] = GETELEMENT21(A,myFinalRow,b*LoNum+myThreadCol,K);//慢
            sB[thId] = GETELEMENT21(B,b*LoNum+myThreadRow,myFinalCol,N);

            __syncthreads();
            
            //矩阵小块(LoNum,LoNum)乘 每个线程算C内[thId/L,thId%L]处的最后值
            float* mysA = &GETELEMENT21(sA,myThreadRow,0,LoNum);//sA第myTR行，sB第myTC列
            float* mysB = &GETELEMENT21(sB,0,myThreadCol,LoNum);

            #pragma unroll 
            for(int i=0;i<LoNum;i++){ //极慢，三倍
                myCresult += *(mysA+i) * *(mysB+i*LoNum); 
            }
        }
    }

    GETELEMENT21(C,myFinalRow,myFinalCol,N) = myCresult; 
    
    
}


int main(int argc, char **argv){
    int device_row_offset=T/LoNum/DEVICEDIM;
    //测试part是否太大
    if(T/LoNum/DEVICEDIM/PART<=0){
        printf("PART error! too many parts!\n");
        return;
    }

    TGPUplan      plan[DEVICEDIM];
    for(int i=0;i<DEVICEDIM;i++){
        cudaSetDevice(i);
        cudaStreamCreate(&plan[i].stream);
    }

    //统一内存h_A,h_B;
    float *h_A = (float *)malloc(sizeof(float)*T*T);
    float *h_B = (float *)malloc(sizeof(float)*T*T);
    
    //给A,B赋值
    if(MATRIXNOR) getNormMatrix(h_A,h_B);
    if(MATRIXEXP){
        getDecayMatrixExp(h_A,1,0.9,M,K);
        getDecayMatrixExp(h_B,1,0.9,K,N);
    }
    if(MATRIXALG){
        getDecayMatrixAlg(h_A,0.1,0.1,K,N);
        getDecayMatrixAlg(h_B,0.1,0.1,K,N);
    }
    // printf("---A---\n");MATRIXSHOW21D(h_A,M,K);

    for(int i=0;i<DEVICEDIM;i++){
        //给私有的bitmap和C分配空间，C用UM
        cudaSetDevice(i);
        cudaMallocManaged((void **)&plan[i].h_A, sizeof(float)*T*T);
        cudaMallocManaged((void **)&plan[i].h_B, sizeof(float)*T*T);
        cudaMallocManaged((void **)&plan[i].h_C, sizeof(float)*T*T);
        cudaMallocManaged((void **)&plan[i].A_normmap, sizeof(float)*(T/LoNum)*(T/LoNum));
        cudaMallocManaged((void **)&plan[i].B_normmap, sizeof(float)*(T/LoNum)*(T/LoNum));

        //UM指导
        cudaMemPrefetchAsync(plan[i].h_A, sizeof(float)*T*T, i);
        cudaMemPrefetchAsync(plan[i].h_B, sizeof(float)*T*T, i);
        cudaMemPrefetchAsync(plan[i].h_C, sizeof(float)*T*T, i);
        cudaMemAdvise(plan[i].h_A, sizeof(float)*T*T, cudaMemAdviseSetReadMostly, i);
        cudaMemAdvise(plan[i].h_B, sizeof(float)*T*T, cudaMemAdviseSetReadMostly, i);

        //流
        cudaStreamCreate(&plan[i].stream);

        //拷贝数据
        cudaMemcpy(plan[i].h_A,h_A,sizeof(float)*T*T,cudaMemcpyHostToDevice);
        cudaMemcpy(plan[i].h_B,h_B,sizeof(float)*T*T,cudaMemcpyHostToDevice);
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
                unroll_get_Fnorm<<<B_blocks/PART,F_threads/8,0,plan[device].stream>>>(plan[device].h_B,plan[device].B_normmap,M,K,p*partBlockOffset);
                // if(p!=PART-1) cudaStreamSynchronize(plan[device].stream);
            }

            //计算某几行A范数和C结果
            int C_blocks = M*N/(LoNum*LoNum),C_threads=LoNum*LoNum;
            for(int p=0;p<PART;p++){
                unroll_get_Fnorm<<<A_blocks/DEVICEDIM/PART,F_threads/8,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,M,K,device*(T/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));

                cudaStreamSynchronize(plan[device].stream);

                get_C_Threads1Element_Mul<<<C_blocks/DEVICEDIM/PART,C_threads,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,plan[device].h_B,plan[device].B_normmap,plan[device].h_C,device*(T/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
            }
        }

        //host同步
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

    
    //检验结果
    if(CHECK) {
        //整合最终C的结果
        float* result_C;
        cudaMallocManaged((void **)&result_C, sizeof(float)*T*T);
        for(int i=0;i<T;i++){
            for(int j=0;j<T;j++){
                result_C[i*T+j]=plan[i/(T/DEVICEDIM)].h_C[i*T+j];
            }
        }
        // check_simple_matrix_mul(h_A,h_B,result_C);
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
