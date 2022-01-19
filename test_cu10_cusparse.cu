#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <assert.h>
#include "tool.cu"

using namespace std;
/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
*/

/*
 * M = # of rows
 * N = # of columns
 */

void print_partial_matrix(float *M, int nrows, int ncols, int max_row,
         int max_col)
 {
     int row, col;
 
     for (row = 0; row < max_row; row++)
     {
         for (col = 0; col < max_col; col++)
         {
             printf("%2.2f ", M[row * ncols + col]);
         }
         printf("...\n");
     }
     printf("...\n");
 }
 
 
double test_cusparse_real(int M, int N, mytype thresh, mytype* origin, mytype* dOri)
{
     mytype *A, *dA, *dB;
     mytype *C, *dC;
     int *dANnzPerRow;
     mytype *dCsrValA;
     int *dCsrRowPtrA;
     int *dCsrColIndA;
     int *dBNnzPerRow;
     mytype *dCsrValB;
     int *dCsrRowPtrB;
     int *dCsrColIndB;
     int *dCNnzPerRow;
     mytype *dCsrValC;
     int *dCsrRowPtrC;
     int *dCsrColIndC;
     int totalANnz,totalBNnz;
     mytype alpha = 3.0f;
     mytype beta = 4.0f;
     cusparseHandle_t handle = 0;
     cusparseMatDescr_t Adescr = 0;
     cusparseMatDescr_t Bdescr = 0;
     cusparseMatDescr_t Cdescr = 0;
     cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;
    cusparseStatus_t stat = CUSPARSE_STATUS_SUCCESS;
 
    // Generate input
    A = (mytype *)malloc(sizeof(mytype) * M * M);
    fill(A, A + M * M, 0.0);
    int trueANnz = getMatrixFromMTX(A, M, M, "../matrix/decay1024_3.mtx", thresh);
    cout << "ratio of non-zero: " << trueANnz << " / " << M * M << " = " << (double)(trueANnz)/(double)(M * M) << endl;
    C = (mytype *)malloc(sizeof(mytype) * M * M);
 
    // Create the cuSPARSE handle
    cusparseCreate(&handle);
 
    // Allocate device memory for vectors and the dense form of the matrix A
    cudaMalloc((void **)&dA, sizeof(mytype) * M * N);
    cudaMalloc((void **)&dB, sizeof(mytype) * M * N);
    cudaMalloc((void **)&dC, sizeof(mytype) * M * M);
    cudaMalloc((void **)&dANnzPerRow, sizeof(int) * M);
    cudaMalloc((void **)&dBNnzPerRow, sizeof(int) * M);
    assert(cudaGetLastError() == cudaSuccess);
 
    // Construct a descriptor of the matrix A
    cusparseCreateMatDescr(&Adescr);
    cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&Bdescr);
    cusparseSetMatType(Bdescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(Bdescr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&Cdescr);
    cusparseSetMatType(Cdescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(Cdescr, CUSPARSE_INDEX_BASE_ZERO);
 
    // Transfer the input vectors and dense matrix A to the device
    cudaMemcpy(dA, A, sizeof(mytype) * M * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, A, sizeof(mytype) * M * M, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0x00, sizeof(mytype) * M * M);
    assert(cudaGetLastError() == cudaSuccess);
 
    // Compute the number of non-zero elements in A
    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, M, Adescr,
                                dA, M, dANnzPerRow, &totalANnz);
    cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, M, Bdescr,
                                    dB, M, dBNnzPerRow, &totalBNnz);
 
    if (totalANnz != trueANnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value: expected %d but got %d\n", trueANnz, totalANnz);
        return 1;
    }
 
    // Allocate device memory to store the sparse CSR representation of A
    cudaMalloc((void **)&dCsrValA, sizeof(mytype) * totalANnz);
    cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1));
    cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalANnz);
    cudaMalloc((void **)&dCsrValB, sizeof(mytype) * totalBNnz);
    cudaMalloc((void **)&dCsrRowPtrB, sizeof(int) * (M + 1));
    cudaMalloc((void **)&dCsrColIndB, sizeof(int) * totalBNnz);
    assert(cudaGetLastError() == cudaSuccess);

    stat = cusparseSdense2csr(handle, M, M, Adescr, dA, M, dANnzPerRow,
        dCsrValA, dCsrRowPtrA, dCsrColIndA);
    stat = cusparseSdense2csr(handle, M, M, Bdescr, dB, M, dBNnzPerRow,
        dCsrValB, dCsrRowPtrB, dCsrColIndB);
    assert(stat == CUSPARSE_STATUS_SUCCESS);

     int nnzC;
     int* nnzTotalDevHostPtr = &nnzC;
     cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
     cudaMalloc((void**)&dCsrRowPtrC, sizeof(int) * (M + 1));

     cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
	cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

	 cusparseXcsrgemmNnz(handle, transA, transB, M, M, M,
		Adescr, trueANnz, dCsrRowPtrA, dCsrColIndA,
		Bdescr, trueANnz, dCsrRowPtrB, dCsrColIndB,
		Cdescr, dCsrRowPtrC, nnzTotalDevHostPtr);
    
    if (NULL != nnzTotalDevHostPtr) {
        nnzC = *nnzTotalDevHostPtr;
            // printf("hahaha\n");
    }
    printf("nnzC = %d\n",nnzC);

    cudaMalloc((void **)&dCsrValC, sizeof(mytype) * nnzC);
    cudaMalloc((void **)&dCsrRowPtrC, sizeof(int) * (M + 1));
    cudaMalloc((void **)&dCsrColIndC, sizeof(int) * nnzC);
    assert(cudaGetLastError() == cudaSuccess);
 
    
    // printf("\n***计时***\n");
    
    printf("start gemm\n");

    for(int i=0; i < WARMUP+TESTTIME; i++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        stat = cusparseScsrgemm(handle, transA, transB, M, M, M,
            Adescr, totalANnz,
            dCsrValA, dCsrRowPtrA, dCsrColIndA,
            Bdescr, totalBNnz,
            dCsrValB, dCsrRowPtrB, dCsrColIndB,
            Cdescr,
            dCsrValC, dCsrRowPtrC, dCsrColIndC);
        assert(stat == CUSPARSE_STATUS_SUCCESS);

        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i>2) sum += elapsed; 
    }
    double cusparsetime=sum/((TESTTIME));
    // printf("cuSPARSE N=%d nnzA=%d(nnz rate=%f) nnzC=%d(nnz rate=%f) \n平均执行时间=%fs\n",M,totalANnz,(double)totalANnz/M/M,nnzC,(double)nnzC/M/M,cusparsetime);
    printf("cuSPARSE M=%d nnzA=%d(nnz rate=%f) \n平均执行时间=%fs\n",M,totalANnz,(double)totalANnz/M/M,cusparsetime);

    stat = cusparseScsr2dense(handle, M, M, Cdescr, dCsrValC, dCsrRowPtrC, dCsrColIndC, dC, M);
    assert(stat == CUSPARSE_STATUS_SUCCESS);

    // Copy the result vector back to the host
    cudaMemcpy(C, dC, sizeof(mytype) * M * M, cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);
 
    printf("C:\n");
    print_partial_matrix(C, M, M, 10, 10);

    // -------------------------------------------------------------------------
    printf("perform gemm in dense format\n");
    // perform matmul in dense format
    mytype* hC2 = (mytype *)malloc(sizeof(mytype) * M * M);
    mytype* dC2;

    cudaMalloc((void**) &dC2, M * M * sizeof(mytype));

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    #if !USINGHALF
    // cublasXtSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, AA, k, BB, n, beta, C, m);
    CHECK_CUBLAS( cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, &alpha, dOri, M, dOri, M, &beta, dC2, M) ) 
    #else
    CHECK_CUBLAS( cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, &alpha, dOri, M, dOri, M, &beta, dC2, M) )
    #endif

    cudaMemcpy(hC2, dC2, M * M * sizeof(mytype), cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);

    // --------------------------------------------------------------------------
    // check hC with hC2
    COUNTERRTRANS(hC2, C, M, M)


 
     free(A);
     free(C);
 
     cudaFree(dA);
     cudaFree(dC);
     cudaFree(dANnzPerRow);
     cudaFree(dCsrValA);
     cudaFree(dCsrRowPtrA);
     cudaFree(dCsrColIndA);
    assert(cudaGetLastError() == cudaSuccess);
 
     cusparseDestroyMatDescr(Adescr);
     cusparseDestroy(handle);
 
     return cusparsetime;
}

int main(int argc, char **argv) {
    // vector<int> sz_vec{1024, 1024, 1024, 8192, 8192, 8192};
    // vector<mytype> tr_vec{0.036, 0.038, 0.04, 0.031, 0.033, 0.039};
    vector<mytype> tr_vec{1e-10, 1e-8, 1e-6, 1e-4, 1e-2};

    int M = 1024;
    mytype* origin = (mytype*) malloc(sizeof(mytype) * M * M);
    getMatrixFromMTX(origin, M, M, "../matrix/decay1024_3.mtx", 0);
    mytype* dOri;
    CHECK_CUDA( cudaMalloc((void**) &dOri, M * M * sizeof(mytype))   )
    cudaMemcpy(dOri, origin, sizeof(mytype) * M * M, cudaMemcpyHostToDevice);


    ofstream out("cusparse_real.csv");
    for (int i = 1; i < 4; i ++) {

        cout << (mytype)tr_vec[i] << " start " << endl;

        double cost = test_cusparse_real(M, M, tr_vec[i],origin, dOri);
        cout << (mytype)tr_vec[i] << " cost_time: " << cost << endl << endl;
        out << (mytype)tr_vec[i] << ", " << cost << endl;
    }
    return 0;
}
