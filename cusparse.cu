#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <cuda.h>
#include "main.h"
/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
*/

/*
 * M = # of rows
 * N = # of columns
 */
 
 //生成25%稠密矩阵
 int generate_random_dense_matrix(int M, int N, float **outA)
 {
     int i, j;
     double rMax = (double)RAND_MAX;
     float *A = (float *)malloc(sizeof(float) * M * N);
     int totalNnz = 0;
 
     for (j = 0; j < N; j++)
     {
         for (i = 0; i < M; i++)
         {
            
             float *curr = A + (j * M + i);
 
             if (i >= 0.25*M)
             {
                 *curr = 0.0f;
             }
             else
             {
                int r = rand( );
                 double dr = (double)r;
                 *curr = (dr / rMax) * 100.0;
             }
 
             if (*curr != 0.0f)
             {
                 totalNnz++;
             }
         }
     }
 
     *outA = A;
     return totalNnz;
 }
 
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
 
 int main(int argc, char **argv)
 {
     float *A, *dA;
     float *B, *dB;
     float *C, *dC;
     int *dANnzPerRow;
     float *dCsrValA;
     int *dCsrRowPtrA;
     int *dCsrColIndA;
     int *dCNnzPerRow;
     float *dCsrValC;
     int *dCsrRowPtrC;
     int *dCsrColIndC;
     int totalANnz;
     float alpha = 3.0f;
     float beta = 4.0f;
     cusparseHandle_t handle = 0;
     cusparseMatDescr_t Adescr = 0;
     cusparseMatDescr_t Cdescr = 0;
     cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;
 
     // Generate input
     srand(9384);
     int trueANnz = generate_random_dense_matrix(M, N, &A);
     int trueBNnz = generate_random_dense_matrix(N, M, &B);
     C = (float *)malloc(sizeof(float) * M * M);
 
     // Create the cuSPARSE handle
     CHECK_CUSPARSE(cusparseCreate(&handle));
 
     // Allocate device memory for vectors and the dense form of the matrix A
     cudaMalloc((void **)&dA, sizeof(float) * M * N);
     cudaMalloc((void **)&dB, sizeof(float) * N * M);
     cudaMalloc((void **)&dC, sizeof(float) * M * M);
     cudaMalloc((void **)&dANnzPerRow, sizeof(int) * M);
 
     // Construct a descriptor of the matrix A
     cusparseCreateMatDescr(&Adescr);
     cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL);
     cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO);

     CHECK_CUSPARSE(cusparseCreateMatDescr(&Cdescr));
     CHECK_CUSPARSE(cusparseSetMatType(Cdescr, CUSPARSE_MATRIX_TYPE_GENERAL));
     CHECK_CUSPARSE(cusparseSetMatIndexBase(Cdescr, CUSPARSE_INDEX_BASE_ZERO));
 
     // Transfer the input vectors and dense matrix A to the device
     cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice);
     cudaMemcpy(dB, B, sizeof(float) * N * M, cudaMemcpyHostToDevice);
     cudaMemset(dC, 0x00, sizeof(float) * M * M);
 
     // Compute the number of non-zero elements in A
     CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, Adescr,
                                 dA, M, dANnzPerRow, &totalANnz));
 
     if (totalANnz != trueANnz)
     {
         fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                 "value: expected %d but got %d\n", trueANnz, totalANnz);
         return 1;
     }
 
     // Allocate device memory to store the sparse CSR representation of A
     cudaMalloc((void **)&dCsrValA, sizeof(float) * totalANnz);
     cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1));
     cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalANnz);
     CHECK_CUSPARSE(cusparseSdense2csr(handle, M, N, Adescr, dA, M, dANnzPerRow,
        dCsrValA, dCsrRowPtrA, dCsrColIndA));

     int nnzC;
     int* nnzTotalDevHostPtr = &nnzC;
     cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
     cudaMalloc((void**)&dCsrRowPtrC, sizeof(int) * (M + 1));

     cusparseOperation_t transA = CUSPARSE_OPERATION_TRANSPOSE;
	cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
	 cusparseXcsrgemmNnz(handle, transA, transB, M, N, K,
		Adescr, trueANnz, dCsrRowPtrA, dCsrColIndA,
		Adescr, trueANnz, dCsrRowPtrA, dCsrColIndA,
		Cdescr, dCsrRowPtrC, nnzTotalDevHostPtr);
        if (NULL != nnzTotalDevHostPtr) {
            nnzC = *nnzTotalDevHostPtr;
            // printf("hahaha\n");
        }
        // printf("%d\n",nnzC);

        cudaMalloc((void **)&dCsrValC, sizeof(float) * nnzC);
        cudaMalloc((void **)&dCsrRowPtrC, sizeof(int) * (M + 1));
        cudaMalloc((void **)&dCsrColIndC, sizeof(int) * nnzC);
 
     
 
    // // Convert A from a dense formatting to a CSR formatting, using the GPU
    // CHECK_CUSPARSE(cusparseSdense2csr(handle, M, N, Adescr, dA, M, dANnzPerRow,
    //     dCsrValA, dCsrRowPtrA, dCsrColIndA));
    // printf("\n***计时***\n");
    
    for(int i=0;i<10;i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        

    //     // CHECK_CUSPARSE(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M,
    //     //     M, N, totalANnz, &alpha, Adescr, dCsrValA,
    //     //     dCsrRowPtrA, dCsrColIndA, dB, N, &beta, dC,
    //     //     // M));

        cusparseScsrgemm(handle, transA, transB, M, N, K,
            Adescr, totalANnz,
            dCsrValA, dCsrRowPtrA, dCsrColIndA,
            Adescr, totalANnz,
            dCsrValA, dCsrRowPtrA, dCsrColIndA,
            Cdescr,
            dCsrValC, dCsrRowPtrC, dCsrColIndC);

        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i>2) sum += elapsed; 
    }
    double cusparsetime=sum/((7));
    // printf("cuSPARSE N=%d nnzA=%d(nnz rate=%f) nnzC=%d(nnz rate=%f) \n平均执行时间=%fs\n",N,totalANnz,(double)totalANnz/M/M,nnzC,(double)nnzC/M/M,cusparsetime);
    printf("cuSPARSE N=%d nnzA=%d(nnz rate=%f) \n平均执行时间=%fs\n",N,totalANnz,(double)totalANnz/M/M,cusparsetime);
 
    //  // Copy the result vector back to the host
    //  CHECK(cudaMemcpy(C, dC, sizeof(float) * M * M, cudaMemcpyDeviceToHost));
 
    // //  printf("C:\n");
    // //  print_partial_matrix(C, M, M, 10, 10);
 
    //  free(A);
    //  free(B);
    //  free(C);
 
    //  CHECK(cudaFree(dA));
    //  CHECK(cudaFree(dB));
    //  CHECK(cudaFree(dC));
    //  CHECK(cudaFree(dANnzPerRow));
    //  CHECK(cudaFree(dCsrValA));
    //  CHECK(cudaFree(dCsrRowPtrA));
    //  CHECK(cudaFree(dCsrColIndA));
 
    //  CHECK_CUSPARSE(cusparseDestroyMatDescr(Adescr));
    //  CHECK_CUSPARSE(cusparseDestroy(handle));
 
     return 0;
}