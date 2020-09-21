#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define loRow 4 //不能是1???????????????????????????
#define loCol 4 //最小的运算单位

#define GETELEMENT21(name,i,j,total_col) name[(i)*(total_col)+j]

#define MATRIXOFFSETCPY(dst,src,size_row,size_col,off_row,off_col,total_col) \
for(int i=0;i<size_row;i++){ \
    for(int j=0;j<size_col;j++){ \
        dst[i][j]=GETELEMENT21(src,off_row+i,off_col+j,total_col); \
    } \
}

#define MATRIXSHOW2D(name,row,col) \
for(int i=0;i<row;i++){ \
    for(int j=0;j<col;j++){ \
        printf("%f ",name[i][j]); \
    } \
    printf("\n"); \
} \
printf("\n");

#define MATRIXSHOW21D(name,row,col) \
for(int i=0;i<row;i++){ \
    for(int j=0;j<col;j++){ \
        printf("%f ",name[(i)*(col)+j]); \
    } \
    printf("\n"); \
} \
printf("\n");

int M=4,K=4,N=4;//输入矩阵的行列数

#define SHAREFLAG 1
//每个kernel获得一个C块
__global__ void naive_kernel(float *d_A, float *d_B, float *d_C, int M,int K,int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int kId = blockIdx.x;//kernel号
    int thId = threadIdx.x;
    //float  th_C[loRow][loRow], th_A[loRow][loCol], th_B[loCol][loRow];
    //__shared__ float s_C[loRow][loRow];
    float a[1];
    a[0]=0.0f;
    // int k_i = kId / (N/loRow);//kernel负责算的C块坐标 (k_i,k_j)
    // int k_j = kId % (N/loRow);
    
    bool valid = kId < (M/loRow)*(N/loRow) && thId < (K/loCol); //合法判断 目标C块有效 且 线程有需要算的
    // if(kId==7) 
    //     printf("kId=%d thId=%d valid=%d C seg=(%d,%d) %d %d %d\n",kId,thId,valid,k_i,k_j,M/loRow,N/loRow,K/loCol);

    //thread算块A(k_i,thId)*B(thId,k_j) 不能有printf??????????????????????????????????
    //a[0]=1+a[1];
    //d_C[0] = a[0]+1;
    printf("%f\n",a[0]+1);
    // d_C[1] = th_C[0][1];
    // d_C[2] = th_C[0][2];
    // d_C[3] = th_C[0][3];
    // d_C[4] = th_C[1][0];
    // d_C[5] = th_C[1][1];
    // d_C[6] = th_C[1][2];
    // d_C[7] = th_C[1][3];
    //MATRIXSHOW2D(th_C,loRow,loRow);

    //reduce 累加K_C 到d_C块(k_i,k_j)  !
    //reduce 同步
    // __syncthreads(); 
    // for(int i=0;i<loRow;i++){
    //     for(int j=0;j<loRow;j++){
    //         for(int t=0;t<blockDim.x;t++){
    //             __syncthreads(); 
    //             if(t==thId && valid){
    //                 if(!SHAREFLAG){
    //                     GETELEMENT21(d_C,k_i*loRow+i,k_j*loRow+j,N)+=th_C[i][j];
    //                 }
    //                 else{
    //                     s_C[i][j] += 1;//th_C[i][j];
    //                     //MATRIXSHOW2D(s_C,loRow,loRow);
    //                 } 
    //             }
    //         }
    //     }
    // }

    // d_C[8] = s_C[0][0];
    // d_C[9] = s_C[0][1];
    // d_C[10] = s_C[0][2];
    // d_C[11] = s_C[0][3];
    // d_C[12] = s_C[1][0];
    // d_C[13] = s_C[1][1];
    // d_C[14] = s_C[1][2];
    // d_C[15] = s_C[1][3];

    // //如果用share mem，0号回写
    // if(SHAREFLAG && valid && thId == 0){
    //     //MATRIXSHOW2D(s_C,loRow,loRow);
    //     for(int i=0;i<loRow;i++){
    //         for(int j=0;j<loRow;j++){
    //             GETELEMENT21(d_C,k_i*loRow+i,k_j*loRow+j,N) = s_C[i][j];
    //         }
    //     }
    // }
}

int main(int argc, char **argv){
    //获得矩阵
    float h_A[M*K],h_B[K*N],h_C[M*N];
    // for(int i=0;i<M;i++){
    //     for(int j=0;j<K;j++){
    //         h_A[i*K+j]=1.5;
    //     }
    // }
    // for(int i=0;i<K;i++){
    //     for(int j=0;j<N;j++){
    //         h_B[i*N+j]=1;
    //     }
    // }

    // //初始化
    float *d_A,*d_B,*d_C;
    int size_A=M*K, size_B=K*N, size_C=M*N;
    // cudaMalloc((void **) &d_A, size_A*sizeof(float));
    // cudaMalloc((void **) &d_B, size_B*sizeof(float));
    cudaMalloc((void **) &d_C, size_C*sizeof(float));

    // cudaMemcpy(d_A, h_A, size_A*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, size_B*sizeof(float), cudaMemcpyHostToDevice);  

    // //调用
    // //int blocks=(M*N)/(loRow*loRow),threads=K/loCol;
    int blocks=1,threads=1;
    naive_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C, d_C, size_C*sizeof(float), cudaMemcpyDeviceToHost);

    printf("--- result ---\n");
    MATRIXSHOW21D(h_C,M,N);

    //end
    // cudaFree(d_A);
    // cudaFree(d_B);
    cudaFree(d_C);
}