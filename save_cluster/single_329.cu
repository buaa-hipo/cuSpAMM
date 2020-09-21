//存档
#include "main.h"



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

    A_normmap[kId] = sqrt(sdata[0]);
}

//每个线程算8个元素,同行相邻的! 每个kernel算一个[LoNum][LoNum]的范数
//每个kernel LoNum*LoNum/8个线程
//!虽然有bank conflict，但是是最快的
__global__ void unroll_get_Fnorm(float *A,float *A_normmap,int m,int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ float sdata[LoNum*LoNum];

    int valid;
    int myBlockRow = kId / (CBLMUN);
    int myBlockCol = kId % (CBLMUN);
    int myThreadRow = thId*8 / LoNum;
    int myThreadCol = thId*8 % LoNum;
    int reduceNum = blockDim.x;

    //每个线程取8个
    valid = id*8 > m*n? 0:1;
    if(valid){
        float t1 = GETELEMENT21(A,myBlockRow*LoNum+myThreadRow,myBlockCol*LoNum+myThreadCol,n);
        float t2 = GETELEMENT21(A,myBlockRow*LoNum+myThreadRow,myBlockCol*LoNum+myThreadCol+1,n);
        float t3 = GETELEMENT21(A,myBlockRow*LoNum+myThreadRow,myBlockCol*LoNum+myThreadCol+2,n);
        float t4 = GETELEMENT21(A,myBlockRow*LoNum+myThreadRow,myBlockCol*LoNum+myThreadCol+3,n);
        float t5 = GETELEMENT21(A,myBlockRow*LoNum+myThreadRow,myBlockCol*LoNum+myThreadCol+4,n);
        float t6 = GETELEMENT21(A,myBlockRow*LoNum+myThreadRow,myBlockCol*LoNum+myThreadCol+5,n);
        float t7 = GETELEMENT21(A,myBlockRow*LoNum+myThreadRow,myBlockCol*LoNum+myThreadCol+6,n);
        float t8 = GETELEMENT21(A,myBlockRow*LoNum+myThreadRow,myBlockCol*LoNum+myThreadCol+7,n);
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
    A_normmap[kId] = sqrt(sdata[0]);
}

//每个kernel计算C[LoNum,LoNum]
//静态无分配版本，每个线程一个元素进行计算
//B normmap !列索引 待优化! ; 分配乘法任务待优化
__global__ void get_C_Threads1Element(float* A,float* A_normmap,float* B,float* B_normmap,float* C){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ int sC_bitmap[CBLMUN];//share mem需要初始化！！
    __shared__ float sA[LoNum*LoNum],sB[LoNum*LoNum],sC[LoNum*LoNum];

    int valid,valid_num=0;
    float norm_mul,t,myCresult=0.0f;
    int myBlockRow = kId / (CBLMUN); 
    int myBlockCol = kId % (CBLMUN); //负责计算块坐标C[Brow,Bcol]处的块

    sC[thId]=0;
    
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
        if(sC_bitmap[b]==1){
            //共同加载share A(mBR)行第b个块,B(mBC)列第b个块
            sA[thId] = GETELEMENT21(A,myBlockRow*LoNum+thId/LoNum,b*LoNum+thId%LoNum,K);
            sB[thId] = GETELEMENT21(B,b*LoNum+thId/LoNum,myBlockCol*LoNum+thId%LoNum,N);
            __syncthreads();

            //矩阵小块(LoNum,LoNum)乘 每个线程算C内[thId/L,thId%L]处的最后值
            #pragma unroll
            for(int i=0;i<LoNum;i++){
                sC[thId] += GETELEMENT21(sA,thId/LoNum,i,LoNum) * GETELEMENT21(sB,i,thId%LoNum,LoNum);
            }
            __syncthreads();
        }
    }

    //每个线程写回自己负责的块C[Brow,Bcol]里面的值
    //！有conflict 回避不了
    GETELEMENT21(C,myBlockRow*LoNum+thId/LoNum,myBlockCol*LoNum+thId%LoNum,N) = sC[thId]; 
}


int main(int argc, char **argv){
    int deviceId=1;
    // cudaGetDevice(&deviceId);
    cudaStream_t streams[2];
    if(STREAM){
        for (int i = 0; i < 2; i++) {
            cudaStreamCreate(&streams[i]);
        }
    }

    int A_size=M*K*sizeof(float), B_size=K*N*sizeof(float), C_size=M*N*sizeof(float);
    int A_map_num = M*K/(LoNum*LoNum);
    int B_map_num = K*N/(LoNum*LoNum);
    float *h_A,*h_B,*h_C,*d_A,*d_B,*d_C;
    float *A_normmap,*B_normmap;
    float *g_A_normmap,*g_B_normmap;

    if(UNIMEM){
        //统一内存
        cudaMallocManaged((void **)&h_A, sizeof(float)*M*K);
        cudaMallocManaged((void **)&h_B, sizeof(float)*K*N);
        cudaMallocManaged((void **)&h_C, sizeof(float)*M*N);
        cudaMalloc((void **)&A_normmap, sizeof(float)*A_map_num);
        cudaMalloc((void **)&B_normmap, sizeof(float)*B_map_num); 
        d_A = h_A;
        d_B = h_B;
        d_C = h_C;
    }
    if(PINMEM){
        //锁内存
        cudaHostAlloc((void **)&h_A, A_size, cudaHostAllocDefault);
        cudaHostAlloc((void **)&h_B, B_size, cudaHostAllocDefault);
        cudaHostAlloc((void **)&h_C, C_size, cudaHostAllocDefault);
        cudaMalloc((void **)&d_A, A_size);
        cudaMalloc((void **)&d_B, B_size);
        cudaMalloc((void **)&d_C, C_size);
        cudaMalloc((void **)&A_normmap, sizeof(float)*A_map_num);
        cudaMalloc((void **)&B_normmap, sizeof(float)*B_map_num);
    }
    // printf("---NORM squrt A:---\n"); MATRIXSHOW21D(A_normmap,1,1);

    if(MATRIXNOR) getNormMatrix(h_A,h_B);
    if(MATRIXEXP){
        getDecayMatrixExp(h_A,1,0.9,M,K);
        getDecayMatrixExp(h_B,1,0.9,K,N);
    }
    if(MATRIXALG){
        getDecayMatrixAlg(h_A,1,0.1,K,N);
        getDecayMatrixAlg(h_B,1,0.1,K,N);
    }
    // printf("---A---\n");MATRIXSHOW21D(h_A,M,K);
    
    //预取
    if(UNIMEM){
        cudaMemPrefetchAsync(d_A, A_size, deviceId);
        cudaMemPrefetchAsync(d_B, B_size, deviceId);
        cudaMemPrefetchAsync(d_C, C_size, deviceId);
        cudaMemAdvise(d_A, A_size, cudaMemAdviseSetReadMostly, deviceId);
        cudaMemAdvise(d_B, B_size, cudaMemAdviseSetReadMostly, deviceId);
    }

    //计时部分
    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;

    for(int i=0;i<TESTTIME;i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        if(PINMEM){
            //拷贝数据
            cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);
        }
        
        //计算范数
        int A_blocks = M*K/(LoNum*LoNum),B_blocks = (K*N)/(LoNum*LoNum),F_threads = LoNum*LoNum;
        if(FORM){
            get_Fnorm<<<A_blocks,F_threads>>>(d_A,A_normmap,M,K);
            get_Fnorm<<<B_blocks,F_threads>>>(d_B,B_normmap,K,N);
        }
        if(UNROLLFORM){
            if(STREAM){
                unroll_get_Fnorm<<<A_blocks,F_threads/8,0,streams[0]>>>(d_A,A_normmap,M,K);
                unroll_get_Fnorm<<<B_blocks,F_threads/8,0,streams[1]>>>(d_B,B_normmap,K,N);
            }
            else{
                unroll_get_Fnorm<<<A_blocks,F_threads/8>>>(d_A,A_normmap,M,K);
                unroll_get_Fnorm<<<B_blocks,F_threads/8>>>(d_B,B_normmap,K,N);
            }
        }
        cudaDeviceSynchronize();

        //矩阵乘
        int C_blocks = M*N/(LoNum*LoNum),C_threads=LoNum*LoNum;
        get_C_Threads1Element<<<C_blocks,C_threads>>>(d_A,A_normmap,d_B,B_normmap,d_C);
        cudaDeviceSynchronize();
        // printf("---result C:---\n"); MATRIXSHOW21D(h_C,M,N);
        

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
    // check_simple_matrix_mul(h_A,h_B,h_C);
    // check(h_A,h_B,h_C);
    float h_Amap[M*K/(LoNum*LoNum)],h_Bmap[K*N/(LoNum*LoNum)];
    cudaMemcpy(&h_Amap, A_normmap, sizeof(float)*A_map_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_Bmap, B_normmap, sizeof(float)*B_map_num, cudaMemcpyDeviceToHost);
    countValid(h_Amap,h_Bmap);
    // printf("---NORM squrt A:---\n"); MATRIXSHOW21D(A_normmap,CBLMUN,CBLMUN);
    // printf("---NORM squrt B:---\n"); MATRIXSHOW21D(B_normmap,CBLMUN,CBLMUN);
    // printf("!!! NORM mul setting = %f!!!\n\n",Norm);
    
    //end
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
