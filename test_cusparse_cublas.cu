#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <assert.h>

using namespace std;

// #include "main.h"
/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
*/

/*
 * M = # of rows
 * N = # of columns
 */

void myTranspose (mytype* A, mytype* B, int m, int n) {
    #pragma omp parallel
    for (int i = 0; i < m; i ++) {
        for (int j = 0; j < n; j ++) {
            B[i * m + j] = A[j * n + i];
        }
    }
}
 
double test_cusparse_cublas(int M, float thresh)
{
    const mytype alf = 3.0f;
    const mytype bet = 4.0f;
    const mytype *alpha = &alf;
    const mytype *beta = &bet;
    
    mytype *hA, *dA;
    mytype *hB, *dB;
    mytype* dC;
    hA = (mytype*)malloc(sizeof(mytype) * M * M);
    hB = (mytype*)malloc(sizeof(mytype) * M * M);

    int A_nnz = getDecayMatrixAlg(hA, 0.1, 0.1, M, M, thresh);
    cout << "ratio of non-zero: " << A_nnz << " / " << M * M << " = " << (double)(A_nnz)/(double)(M * M) << endl;

    myTranspose(hA, hB, M, M);

    cudaMalloc((void **)&dA, sizeof(mytype)*M*M);
    cudaMalloc((void **)&dB, sizeof(mytype)*M*M);

    cudaMemcpy(dA, hA, sizeof(mytype) * M * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(mytype) * M * M, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dC, sizeof(mytype)*M*M);
    
    // cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);


    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;

    for(int i=0; i < TESTTIME + WARMUP; i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        #if !USINGHALF
        // cublasXtSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, AA, k, BB, n, beta, C, m);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, M, M, alpha, dA, M, dB, M, beta, dC, M); 
        #else
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, M, M, alpha, dA, M, dB, M, beta, dC, M); 
        #endif
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i>=WARMUP) sum += elapsed;
    }
    // printf("cuBLAS平均执行时间=%fs\n",sum/(TESTTIME-WARMUP));
    return (double)(sum/(TESTTIME));
}


int main(int argc, char **argv) {
    vector<int> sz_vec{1024, 1024, 1024, 8192, 8192, 8192};
    vector<float> tr_vec{0.0378, 0.0412, 0.0445, 0.0352, 0.0415, 0.044};
    ofstream out("cusparse_cublas_time.csv");
    for (int i = 0; i < 6; i ++) {
        double cost = test_cusparse_cublas(sz_vec[i], tr_vec[i]);
        cout << sz_vec[i] << ", " << tr_vec[i] << ": " << cost << endl;
        out << sz_vec[i] << ", " << tr_vec[i] << ", " << cost << endl;
    }
    return 0;
}