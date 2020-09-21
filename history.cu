//每个线程算8个元素,同行相邻的! 每个kernel算一个[LoNum][LoNum]的范数
//每个kernel LoNum*LoNum/8个线程
//!虽然有bank conflict，但是是最快的
__global__ void unroll_get_Fnorm(const float* __restrict__ A,float *A_normmap,int m,int n,int blockRowOff){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ float sdata[LoNum*LoNum/8];

    int valid;
    const int myBlockRow = kId / (CBLMUN)+blockRowOff;
    const int myBlockCol = kId % (CBLMUN);
    const int myBlockId = myBlockRow*(T/LoNum)+myBlockCol;
    const int myThreadRow = thId / 4;
    const int myThreadCol = thId % 4;
    const int myFinalRow = myBlockRow*LoNum+myThreadRow;
    const int myFinalCol = myBlockCol*LoNum+myThreadCol*8;
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
    
    float val;
    //warp内reduce
    if (thId < 32)
    {
        volatile float* sw = sdata;
        // printf("sw[%d] %f\n",thId,sw[thId]);
        sw[thId] += sw[thId + 32];
        sw[thId] += sw[thId + 16];
        sw[thId] += sw[thId + 8];
        sw[thId] += sw[thId + 4];
        sw[thId] += sw[thId + 2];
        sw[thId] += sw[thId + 1];
    }
    if(thId==0){
        A_normmap[myBlockId] = sqrt(sdata[0]);
        // printf("shared mem kid=%d val=%f\n",kId,sdata[0]);
    } 
}


//20200902 18:58
//4个warp，计算32*32 (4个warp 4*32个线程)
__global__ void get_C_FP16_B32(const half* __restrict__ A,const float* __restrict__ A_normmap,const half* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;
    int thId = threadIdx.x;
    int warpId = thId/32;
    __shared__ int sC_bitmap[(CBLMUN/4+1)*4]; //四字节对齐
    __shared__ int sC_offset[(CBLMUN/4+1)*4];
    __shared__ half st[LoNum*LoNum];
    __shared__ half sA0[LoNum*LoNum],sB0[LoNum*LoNum]; 
    __shared__ half sA1[LoNum*LoNum],sB1[LoNum*LoNum]; 

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
    for(int i=thId;i<(CBLMUN/4+1)*4;i+=blockDim.x){
        if(i<(CBLMUN)){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow,i,CBLMUN) * GETELEMENT21(B_normmap,i,myBlockCol,CBLMUN);
            sC_bitmap[i] = norm_mul>Norm? 1:0; //!范数计算有E的浮动误差，应该是6位有效数字
        //    printf("thid=%d i=%d sC_bitmap=%d\n",thId,i,sC_bitmap[i]);
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
    //reduce算一共有几个非零值
    for (unsigned int s = CBLMUN / 2; s > 0; s >>= 1) {
		if (thId < s) {
			sC_bitmap[thId] += sC_bitmap[thId + s];
		}
		__syncthreads();
    }
    const int validNum = sC_bitmap[0]; 
    // if(thId==0) printf("validNum=%d\n",sC_bitmap[0]);
    
    int warpi=warpId/2;
    int warpj=warpId%2;
    const int myFinalRow16 = myBlockRow*2+warpi;
    const int myFinalCol16 = myBlockCol*2+warpj;

    int this_b,next_b;
    half * A_this_read=sA0;
    half * B_this_read=sB0;
    half * A_this_write=sA1;
    half * B_this_write=sB1;
    if(validNum>0){
        this_b=sC_offset[0];
        // 4*32线程并行加载32*32的A，B，每个warp 8行
        #pragma unroll
        for(int line=warpId*8;line<(warpId+1)*8;line++){
            GETELEMENT21(A_this_read,line,thId%32,LoNum) = GETELEMENT21(A,myBlockRowOff+line,this_b*LoNum+thId%32,K);
            GETELEMENT21(B_this_read,line,thId%32,LoNum) = GETELEMENT21(B,this_b*LoNum+line,myBlockColOff+thId%32,K);
        }
    }

    //遍历bitmap,每个线程负责一个位置的元素
    #pragma unroll 
    for(int i=0;i<validNum;i++){
        __syncthreads();
        this_b = sC_offset[i];
        if(i+1<validNum){
            next_b = sC_offset[i+1];
            #pragma unroll
            for(int line=warpId*8;line<(warpId+1)*8;line++){
                GETELEMENT21(A_this_write,line,thId%32,LoNum) = GETELEMENT21(A,myBlockRowOff+line,next_b*LoNum+thId%32,K);
                GETELEMENT21(B_this_write,line,thId%32,LoNum) = GETELEMENT21(B,next_b*LoNum+line,myBlockColOff+thId%32,K);
            }
        }
        
        for(int k=0;k<2;k++){
            wmma::load_matrix_sync(a_frag, GETOFF21(A_this_read,warpi*16,k*16,LoNum), LoNum);
            wmma::load_matrix_sync(b_frag, GETOFF21(B_this_read,k*16,warpj*16,LoNum), LoNum);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
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

    wmma::store_matrix_sync(GETOFF21(C,myFinalRow16*16,myFinalCol16*16,K), c_frag, K,wmma::mem_row_major);
}

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



wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a00_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a01_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b00_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b01_frag;

wmma::load_matrix_sync(a00_frag, GETOFF21(A,myFinalRow16*16,(this_b*LoNum/16+0)*16,K), K);
        wmma::load_matrix_sync(b00_frag, GETOFF21(B,(this_b*LoNum/16+0)*16,myFinalCol16*16,K), K);
        wmma::mma_sync(c_frag, a00_frag, b00_frag, c_frag);

        wmma::load_matrix_sync(a01_frag, GETOFF21(A,myFinalRow16*16,(this_b*LoNum/16+1)*16,K), K);
        wmma::load_matrix_sync(b01_frag, GETOFF21(B,(this_b*LoNum/16+1)*16,myFinalCol16*16,K), K);
        wmma::mma_sync(c_frag, a01_frag, b01_frag, c_frag);

__global__ void get_C_Threads1Element_Mul(const float* __restrict__ A,const float* __restrict__ A_normmap,const float* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset){
    const int CBL = CBLMUN;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ int sC_bitmap[CBLMUN];//share mem需要初始化！！
    // __shared__ int sC_bitmap_debug[CBLMUN];
    __shared__ int sC_offset[CBLMUN];
    __shared__ float sA0[LoNum*LoNum],sB0[LoNum*LoNum]; //sC可以换成局部变量，但有local的风险
    __shared__ float sA1[LoNum*LoNum],sB1[LoNum*LoNum]; 

    float norm_mul,myCresult=0.0f;
    const int myBlockRow = kId / (CBLMUN) + main_row_offset; 
    const int myBlockCol = kId % (CBLMUN); //负责计算块坐标C[Brow,Bcol]处的块
    const int myBlockRowOff = myBlockRow*LoNum;
    const int myBlockColOff = myBlockCol*LoNum;
    const int myThreadRow = thId/LoNum;
    const int myThreadCol = thId%LoNum;
    const int myFinalRow = myBlockRowOff+myThreadRow;
    const int myFinalCol = myBlockColOff+myThreadCol;
    
    
    // 需要A_norm第R行，B_norm第C列
    #pragma unroll
    for(int i=thId;i<CBLMUN;i+=blockDim.x){
        if(thId<(CBLMUN)){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow,i,CBLMUN) * GETELEMENT21(B_normmap,i,myBlockCol,CBLMUN);
            sC_bitmap[i] = norm_mul>Norm? 1:0; //!范数计算有E的浮动误差，应该是6位有效数字
        }
    }
    __syncthreads();//不能和下面合并！因为有的线程的b可能没算完就结束了，但是非常费时间

    // if(thId==0){
    //     curandState state;
    //     for(int i=0;i<CBL;i++){
    //         curand_init(thId, i, 0, &state);
    //         unsigned int x = curand(&state)%2;
    //         // if(x%2==0) sC_bitmap[i]=1;
    //         // else sC_bitmap[i]=0;
    //         sC_bitmap[i]=1;
    //         // printf("%d ",sC_bitmap[i]);
    //     }
    // }

    for(int i=thId;i<CBL;i+=blockDim.x){
        if(sC_bitmap[i]==1){
            int t=0;
            for(int j=0;j<i;j++){
                if(sC_bitmap[j]==1){
                    t++;
                }
            }
            sC_offset[t]=i;
            // printf("id=%d dim=%d t=%d %d\n",thId,blockDim.x,t,sC_offset[t]);
        }
    }
    __syncthreads();
    //reduce算一共有几个非零值
    for (unsigned int s = CBL/2*2 / 2; s > 0; s >>= 1) {
		if (thId < s) {
			sC_bitmap[thId] += sC_bitmap[thId + s];
		}
		__syncthreads();
    }
    const int validNum = sC_bitmap[0]; //不会conflict，只有同bank不同位置才会发生
    // if(thId==0) printf("\nsum=%d\n",sC_bitmap[0]);
    

    //遍历bitmap,每个线程负责一个位置的元素
    //先使用sA0的数据
    int this_b,next_b;
    if(validNum>0){
        this_b=sC_offset[0];
        sA0[thId] = GETELEMENT21(A,myFinalRow,this_b*LoNum+myThreadCol,K);//慢
        sB0[thId] = GETELEMENT21(B,this_b*LoNum+myThreadRow,myFinalCol,N);
    }
    float * A_this_read=sA0;
    float * B_this_read=sB0;
    float * A_this_write=sA1;
    float * B_this_write=sB1;
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
        float* mysA = &GETELEMENT21(A_this_read,myThreadRow,0,LoNum);//sA第myTR行，sB第myTC列
        float* mysB = &GETELEMENT21(B_this_read,0,myThreadCol,LoNum);

        #pragma unroll 
        for(int i=0;i<LoNum;i++){ //极慢，三倍
            myCresult += *(mysA+i) * *(mysB+i*LoNum); 
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

if(thId==0){
    curandState state;
    for(int i=0;i<CBL;i++){
        curand_init(thId, i, 0, &state);
        unsigned int x = curand(&state)%2;
        if(x%2==0) sC_bitmap[i]=1;
        else sC_bitmap[i]=0;
        printf("%d ",sC_bitmap[i]);
    }
}

//每个kernel计算C[LoNum,LoNum]
//静态无分配版本，每个线程一个元素进行计算
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

//使用原语reduce，每个warp自加，然后给前32个再累加
//每个kernel LoNum*LoNum/8个线程,4个warp
__global__ void unroll_get_Fnorm_pri(const float* __restrict__ A,float *A_normmap,int m,int n,int blockRowOff){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel
    int thId = threadIdx.x;
    __shared__ float sdata[4];

    int valid=0;
    const int myBlockRow = kId / (CBLMUN)+blockRowOff;
    const int myBlockCol = kId % (CBLMUN);
    const int myBlockId = myBlockRow*(T/LoNum)+myBlockCol;
    const int myThreadRow = thId / 4;
    const int myThreadCol = thId % 4;
    const int myFinalRow = myBlockRow*LoNum+myThreadRow;
    const int myFinalCol = myBlockCol*LoNum+myThreadCol*8;
    const int reduceNum = blockDim.x;

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
    
    if (thId < 4)
    {
        // printf("thid=%d val=%f sw[thid]=%f\n",thId,val,sdata[thId]);
        val=sdata[thId];
        // printf("%d %f\n",thId,sdata[thId]);
        for (int offset = 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    if(thId==0){
        A_normmap[myBlockId] = sqrt(val); //记得开方
        // printf("pri kid=%d val=%f\n",kId,val);
    } 
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

//1个warp，计算16*16
__global__ void get_C_FP16_B16(const half* __restrict__ A,const float* __restrict__ A_normmap,const half* __restrict__ B,const float* __restrict__ B_normmap,float* C,const int main_row_offset){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;
    int thId = threadIdx.x;
    int warpId = thId/32;
    __shared__ int sC_bitmap[CBLMUN*2];
    
    float norm_mul,myCresult=0.0f;
    const int myBlockRow = kId / (CBLMUN*2) + main_row_offset; 
    const int myBlockCol = kId % (CBLMUN*2); //B=16,负责计算块坐标C[Brow,Bcol]处的块
    const int myBlockRowOff = myBlockRow*16;
    const int myBlockColOff = myBlockCol*16;
    
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    //需要A_norm第R行，B_norm第C列
    #pragma unroll
    for(int i=thId;i<CBLMUN*2;i+=blockDim.x){
        if(thId<(CBLMUN*2)){
            norm_mul = GETELEMENT21(A_normmap,myBlockRow/2,i/2,CBLMUN) * GETELEMENT21(B_normmap,i/2,myBlockCol/2,CBLMUN);
            sC_bitmap[i] = norm_mul>Norm? 1:0; //!范数计算有E的浮动误差，应该是6位有效数字
        }
    }
    __syncthreads();//不能和下面合并！因为有的线程的b可能没算完就结束了，但是非常费时间


    //遍历bitmap,每个线程负责一个位置的元素
    #pragma unroll 
    for(int b=0;b<CBLMUN*2;b++){
        if(sC_bitmap[b]==1){//慢
            //计算大块为C[myBlockRow][myBlockCol]+=A[mR][b]*B[b][mC];
            wmma::load_matrix_sync(a_frag, GETOFF21(A,myBlockRowOff,b*16,K), K);
            wmma::load_matrix_sync(b_frag, GETOFF21(B,b*16,myBlockColOff,K), K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    wmma::store_matrix_sync(GETOFF21(C,myBlockRowOff,myBlockColOff,K), c_frag, K,wmma::mem_row_major);

}