
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
    __shared__ float sdata[LoNum*LoNum/8/32];

    int valid=0;
    const int myBlockRow = kId / (n/LoNum)+blockRowOff;
    const int myBlockCol = kId % (n/LoNum);
    const int myBlockId = myBlockRow*(n/LoNum)+myBlockCol;
    const int myThreadRow = thId / (LoNum/8);
    const int myThreadCol = thId % (LoNum/8);
    const int myFinalRow = myBlockRow*LoNum+myThreadRow;
    const int myFinalCol = myBlockCol*LoNum+myThreadCol*8;

    //每个线程取1个
    float val;
    valid = id > m*n? 0:1;
    if(valid){
        int tadd = myFinalRow*n+myFinalCol;
        const half t1 = A[tadd];
        const half t2 = A[tadd+1];
        const half t3 = A[tadd+2];
        const half t4 = A[tadd+3];
        const half t5 = A[tadd+4];
        const half t6 = A[tadd+5];
        const half t7 = A[tadd+6];
        const half t8 = A[tadd+7];
        const half t11 = __hmul(t1,t1);
        const half t21 = __hmul(t2,t2);
        const half t31 = __hmul(t3,t3);
        const half t41 = __hmul(t4,t4);
        const half t51 = __hmul(t5,t5);
        const half t61 = __hmul(t6,t6);
        const half t71 = __hmul(t7,t7);
        const half t81 = __hmul(t8,t8);

        const half t12 = __hadd(t11,t21);
        const half t23 = __hadd(t12,t31);
        const half t34 = __hadd(t23,t41);
        const half t45 = __hadd(t34,t51);
        const half t56 = __hadd(t45,t61);
        const half t67 = __hadd(t56,t71);
        const half t78 = __hadd(t67,t81);
        val = __half2float(t78);
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


//每个kernel计算C[LoNum,LoNum]
//静态无分配版本，每个线程一个元素进行计算，LoNum*LoNum个线程
__global__ void get_C_Threads1Element_Mul_FP16(const half* __restrict__ A,const float* __restrict__ A_normmap,const half* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset,float Norm){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    int REDUCECBL = 1<<(int)(log2(CBLMUN*1.0)+1);

    __shared__ int sC_bitmap[CBLMUN*2];//share mem需要初始化！！
    // __shared__ int sC_bitmap_debug[CBLMUN];
    __shared__ int sC_offset[CBLMUN];
    __shared__ half sA0[LoNum*LoNum],sB0[LoNum*LoNum]; //sC可以换成局部变量，但有local的风险
    __shared__ half sA1[LoNum*LoNum],sB1[LoNum*LoNum]; 

    float norm_mul;
    half myCresult=__float2half(0.0f);
    const int myBlockRow = kId / (N/LoNum) + main_row_offset; 
    const int myBlockCol = kId % (N/LoNum); //负责计算块坐标C[Brow,Bcol]处的块
    const int myBlockRowOff = myBlockRow*LoNum;
    const int myBlockColOff = myBlockCol*LoNum;
    const int myThreadRow = thId/LoNum;
    const int myThreadCol = thId%LoNum;
    const int myFinalRow = myBlockRowOff+myThreadRow;
    const int myFinalCol = myBlockColOff+myThreadCol;

    // if(thId==0){
    //     printf("kid=%d br=%d bc=%d\n",kId,myBlockRow,myBlockCol);
    // }
    
    
    // 需要A_norm第R行，B_norm第C列
    #pragma unroll
    for(int i=thId;i<REDUCECBL;i+=blockDim.x){
        if(i<(CBLMUN)){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow,i,K/LoNum) * GETELEMENT21(B_normmap,i,myBlockCol,N/LoNum);
            sC_bitmap[i] = norm_mul>Norm? 1:0; //!范数计算有E的浮动误差，应该是6位有效数字
        }
        else{
            sC_bitmap[i] = 0;
        }
    }
    __syncthreads();//不能和下面合并！因为有的线程的b可能没算完就结束了，但是非常费时间


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

    // //reduce算一共有几个非零值,reduce版本只能处理小规模且为2的幂
    for (unsigned int s = REDUCECBL/2; s > 0; s >>= 1) {
		if (thId < s) {
			sC_bitmap[thId] += sC_bitmap[thId + s];
		}
		__syncthreads();
    }
    const int validNum = sC_bitmap[0]; //不会conflict，只有同bank不同位置才会发生
    // // if(kId==0&&thId==0) printf("\nsum=%d\n",sC_bitmap[0]);

    
    // if(kId==0&&thId==0) printf("\nsum=%d\n",validNum);

    //遍历bitmap,每个线程负责一个位置的元素
    //先使用sA0的数据
    int this_b,next_b;
    if(validNum>0){
        this_b=sC_offset[0];
        sA0[thId] = GETELEMENT21(A,myFinalRow,this_b*LoNum+myThreadCol,K);//慢
        sB0[thId] = GETELEMENT21(B,this_b*LoNum+myThreadRow,myFinalCol,N);
        // if(kId==1&&thId==0) printf("read %d %d\n",myFinalRow,this_b*LoNum+myThreadCol); 
    }
    half * A_this_read=sA0;
    half * B_this_read=sB0;
    half * A_this_write=sA1;
    half * B_this_write=sB1;
    #pragma unroll 
    for(int i=0;i<validNum;i++){
        __syncthreads(); 
        this_b = sC_offset[i];

        //[计算32*32规模的矩阵乘]
        //共同加载share A(mBR)行第b个块,B(mBC)列第b个块
        if(i<validNum-1){
            next_b = sC_offset[i+1];
            A_this_write[thId] = GETELEMENT21(A,myFinalRow,next_b*LoNum+myThreadCol,K);//慢
            B_this_write[thId] = GETELEMENT21(B,next_b*LoNum+myThreadRow,myFinalCol,N);
        }
        
        //矩阵小块(LoNum,LoNum)乘 每个线程算C内[thId/L,thId%L]处的最后值
        half* mysA = &GETELEMENT21(A_this_read,myThreadRow,0,LoNum);//sA第myTR行，sB第myTC列
        half* mysB = &GETELEMENT21(B_this_read,0,myThreadCol,LoNum);
        
        #pragma unroll
        for(int i=0;i<LoNum;i++){ //极慢，三倍
            // myCresult += *(mysA+i) * *(mysB+i*LoNum); 
            const __half a = *(mysA+i);
            const __half b = *(mysB+i*LoNum);
            myCresult +=__hmul(a,b);
            // if(myFinalRow==1186&&myFinalCol==1183){
            //     printf("kid=%d thid=%d b=%d %f %f %f\n",kId,thId,this_b,myCresult,A[1186*K+i],B[i*N+1183]);
            // } 
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

        //UM指导
        cudaMemPrefetchAsync(plan[i].h_A, sizeof(mytype)*M*K, i); 
        cudaMemPrefetchAsync(plan[i].h_B, sizeof(mytype)*K*N, i);
        cudaMemPrefetchAsync(plan[i].h_C, sizeof(float)*M*N, i);
        cudaMemAdvise(plan[i].h_A, sizeof(mytype)*M*K, cudaMemAdviseSetReadMostly, i);
        cudaMemAdvise(plan[i].h_B, sizeof(mytype)*K*N, cudaMemAdviseSetReadMostly, i);

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
                    unroll_get_Fnorm_pri<<<B_blocks/PART,F_threads/8,0,plan[device].stream>>>(plan[device].h_B,plan[device].B_normmap,K,N,p*partBlockOffset);
                }
                else{

                }
                
                #else
                if(LoNum==32){
                    unroll_get_Fnorm_pri_FP16<<<B_blocks/PART,F_threads/8,0,plan[device].stream>>>(plan[device].h_B,plan[device].B_normmap,K,N,p*partBlockOffset);
                }
                else{

                }
                #endif
            }
            cudaStreamSynchronize(plan[device].stream);
            // printf("---the normmap of B:---\n");
            // // MATRIXSHOW21D(plan[device].B_normmap,B_blocks,1);


            //计算某几行A范数和C结果
            int C_blocks = M*N/(LoNum*LoNum),C_threads=LoNum*LoNum;
            for(int p=0;p<PART;p++){
                #if !USINGHALF
                if(LoNum==32){
                    unroll_get_Fnorm_pri<<<A_blocks/DEVICEDIM/PART,F_threads/8,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,M,K,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                }
                else{

                }
                
                #else
                if(LoNum==32){
                    unroll_get_Fnorm_pri_FP16<<<A_blocks/DEVICEDIM/PART,32*4,0,plan[device].stream>>>(plan[device].h_A,plan[device].A_normmap,M,K,device*(M/LoNum/DEVICEDIM)+p*(partBlockOffset/DEVICEDIM));
                }
                else{

                }
                #endif

                cudaStreamSynchronize(plan[device].stream);

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
