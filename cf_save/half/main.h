
#include <omp.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <unistd.h>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <mma.h>
#include "device_launch_parameters.h"
#include "para.h"
// using namespace std;
using namespace nvcuda;
// #define USINGHALF 0

#if USINGHALF
typedef half mytype;
#else
typedef float mytype;
#endif

#define LoNum 32 //32的倍数

// const double NormINIT=0;
#define TUNINGFLAG 0
const double ExpectedRate = 0.75; 
#define TUNINGTIME 20
#define TUNINGERROR 0.01

// #define TESTTIME 1
// #define WARMUP 0
// const int inM=1<<15;
// const int inK=1<<15;
// const int inN=1<<15;
#define T 32
// #define DEVICEDIM 2
#define PART 1
#define CHECK 1
#define TRUNCATIONNUM 0.030

#define SpAMM 1
#define CUBLAS 1

const int M = ((inM-1)/LoNum+1) * LoNum;
const int K = ((inK-1)/LoNum+1) * LoNum;
const int N = ((inN-1)/LoNum+1) * LoNum;

const float E = 1e-6;
//模式

#define UNIMEM 1  //统一内存，+预取
#define PINMEM 0  //使用锁内存
#define TEXTURE 0 //显示使用纹理


#define FORM 0
#define UNROLLFORM 1
#define STREAM 0 //多机时不能使用单机的流模式

#define WARPSIZE 32

// #define CDIVIDE 0 //每个kernel负责几个块
const int CBLMUN = K/LoNum;
const int CBDIM = LoNum * LoNum;
extern float Norm;


#define GIVE_NUMBER21(dst, src, m, n)                                          \
    for (int i = 0; i < m; i++) {                                              \
        for (int j = 0; j < n; j++) {                                          \
            dst[i][j] = src[i * n + j];                                        \
        }                                                                      \
    }

#define MATRIXSHOW2D(name, row, col)                                           \
    for (int i = 0; i < row; i++) {                                            \
        for (int j = 0; j < col; j++) { \
            float t;\
            if (typeid(name[i][j]) != typeid(t)){\
                t=__half2float(name[i][j]);\
            }                                     \
            else{\
                t=name[i][j];\
            }\
            printf("%f ", t);                                         \
        }                                                                      \
        printf("\n");                                                          \
    }                                                                          \
    printf("\n");

#define MATRIXSHOW2DINT(name, row, col)                                        \
    for (int i = 0; i < row; i++) {                                            \
        for (int j = 0; j < col; j++) {                                        \
            printf("%4d ", name[i][j]);                                        \
        }                                                                      \
        printf("\n");                                                          \
    }                                                                          \
    printf("\n");

#define GETELEMENT21(name, i, j, total_col) name[(i) * (total_col) + j]
#define GETOFF21(name, i, j, total_col) name+((i) * (total_col) + j)
#define TEXTURE_GETELEMENT21(name, i, j, total_col)                            \
    tex1Dfetch(name, (i) * (total_col) + j)

#define MATRIXSHOW21D(name, row, col)                                          \
    for (int i = 0; i < row; i++) {                                            \
        for (int j = 0; j < col; j++) {      \
            float t;\
            mytype tt;\
            if(typeid(tt)!=typeid(t)){\
                t=__half2float(name[(i) * (col) + j]);\
            }                                  \
            else{\
                t=name[(i) * (col) + j];\
            }\
            printf("%f ", t);                              \
        }                                                                      \
        printf("\n");                                                          \
    }                                                                          \
    printf("\n");

void check(float *A, float *B, float *C);
void check_simple_matrix_mul(float *in_A, float *in_B, float *in_C);
void check_simple_gpu(mytype *A, mytype *B, float *in_C);
double run_cublas_time(mytype *A, mytype *B);
void countValid(float *A_normmap, float *B_normmap);
float tuneValidRate(float *A_normmap, float *B_normmap,int m,int n);
void checkNormMap(float *in_A, float *A_normmap);

void getMatrixFromCSV(mytype* A,int m,int n,std::string filename);
void getMatrixFromMTX(mytype* A,int m,int n,std::string filename);
void getNormMatrix(mytype *A, mytype *B);
void getDecayMatrixExp(mytype *A, float c, float v, int m, int n);
void getDecayMatrixAlg(mytype *A, float c, float v, int m, int n);

float *copy_B(mytype *B,int m,int n);
mytype *copy_B2(mytype *B,int m,int n);
void truncation(float *M, float *ORI, float flag);
void getDecayMatrixAlgDouble(double *A, double c, double v, int m, int n);
