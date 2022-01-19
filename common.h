#include <sys/time.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK_CUDA(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        fprintf(stderr, "Error: %s\n", cusparseGetErrorString(err));           \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#define USINGHALF 0
#define TESTTIME 10
#define WARMUP 3

#if USINGHALF
typedef half mytype;
#define cudatatype CUDA_R_16F
#else
typedef float mytype;
#define cudatatype CUDA_R_32F
#endif

//代数衰减 |a_ij| < c/(|i-j|^v+1); c>0, v>0
// 增加了截断
int getDecayMatrixAlg(mytype* A, float c, float v,int m,int n, float thresh){
    float t;
    int nnz = 0;
    #pragma omp parallel
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            t = (float) c / ((float) pow(abs(i-j),v)+1);
            // A[i*n+j] = t;
            if (t < thresh) A[i*n+j] = 0;
            else {
                #if USINGHALF
                A[i*n+j] = __float2half(t);
                #else
                A[i*n+j] = t;//((mytype)(rand()%10))/10*t;
                #endif
                nnz ++;
            }
            // if(rand()%2==0) A[i*n+j]*=-1;
        }
    }
    return nnz;
}

#endif // _COMMON_H