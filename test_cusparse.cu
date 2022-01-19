#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
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

 
double test_cusparse_csr(int M, int N, mytype thresh)
{
    mytype *hA, *dA;

    mytype alpha = 3.0f;
    mytype beta = 4.0f;
    cusparseHandle_t handle = 0;
    cusparseDnMatDescr_t Adescr_dn;
    cusparseSpMatDescr_t Adescr, Bdescr;
    cusparseSpMatDescr_t Cdescr = 0;

    cudaDataType        computeType = cudatatype;

    // Create the cuSPARSE handle
    CHECK_CUSPARSE(cusparseCreate(&handle));
 
//-------------------------------------------------------------------
    // prepare A
    // Generate input
    // srand(9384);
    hA = (mytype *)malloc(sizeof(mytype) * M * M);
    int A_nnz = getDecayMatrixAlg(hA, 0.1, 0.1, M, M, thresh);
    cout << "ratio of non-zero: " << A_nnz << " / " << M * M << " = " << (double)(A_nnz)/(double)(M * M) << endl;
 
    // Allocate device memory for the dense form of the matrix A
    cudaMalloc((void **)&dA, sizeof(mytype) * M * M);
 
    // Transfer the input dense matrix A to the device
    cudaMemcpy(dA, hA, sizeof(mytype) * M * M, cudaMemcpyHostToDevice);


    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&Adescr_dn, M, M, M, dA,
                                        cudatatype, CUSPARSE_ORDER_ROW) )
    int   *dA_csr_offsets, *dA_csr_columns;
    mytype *dA_csr_values;

    // Allocate device memory to store the sparse CSR representation of A
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_offsets, (M + 1) * sizeof(int)) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&Adescr, M, M, 0,
                                      dA_csr_offsets, NULL, NULL, 
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, cudatatype) )

//----------------------------------------------------------------------------
    // dense to csr (cuda v11.2.0 or above required)
    size_t bufferSize0 = 0;
    void* dBuffer0 = NULL;
    CHECK_CUSPARSE (
        cusparseDenseToSparse_bufferSize(handle, 
                                        Adescr_dn, 
                                        Adescr, 
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize0)
    )
    CHECK_CUDA( cudaMalloc(&dBuffer0, bufferSize0) )

    CHECK_CUSPARSE (
        cusparseDenseToSparse_analysis(handle, 
                                        Adescr_dn, 
                                        Adescr, 
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer0)
    )

    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(Adescr, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_columns, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_values,  nnz * sizeof(mytype)) )

    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(Adescr, dA_csr_offsets, dA_csr_columns,
                                           dA_csr_values) )
    // excute dense to sparse conversion
    CHECK_CUSPARSE (
        cusparseDenseToSparse_convert(handle, 
                                        Adescr_dn, 
                                        Adescr, 
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer0)
    )
    
    CHECK_CUSPARSE( cusparseDestroyDnMat(Adescr_dn) )
    CHECK_CUDA( cudaFree(dA) )

//----------------------------------------------------------------------------
    // prepare B
    int   *dB_csr_offsets0, *dB_csr_columns0;
    int   *dB_csr_offsets, *dB_csr_columns;
    mytype *dB_csr_values0, *dB_csr_values;


    // Allocate device memory to store the sparse CSR representation of B
    CHECK_CUDA( cudaMalloc((void**) &dB_csr_offsets0, (M + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_csr_columns0, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dB_csr_values0,  nnz * sizeof(mytype)) )

    CHECK_CUDA( cudaMalloc((void**) &dB_csr_offsets, (M + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_csr_columns, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dB_csr_values,  nnz * sizeof(mytype)) )


    CHECK_CUDA( cudaMemcpy(dB_csr_offsets0, dA_csr_offsets, sizeof(int) * (M + 1), cudaMemcpyDeviceToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_csr_columns0, dA_csr_columns, sizeof(int) * nnz, cudaMemcpyDeviceToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_csr_values0, dA_csr_values, sizeof(mytype) * nnz, cudaMemcpyDeviceToDevice) )

    size_t tp_buffer_size = 0;
    void* tp_buffer = NULL;
    // transpose B (csc format is the transposed B)
    CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(handle, M, M, nnz, 
                                dB_csr_values0,     // nnz
                                dB_csr_offsets0,    // M + 1
                                dB_csr_columns0,    // nnz
                                dB_csr_values,      // nnz
                                dB_csr_offsets,     // M + 1
                                dB_csr_columns,     // nnz
                                cudatatype, 
                                CUSPARSE_ACTION_NUMERIC, 
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUSPARSE_CSR2CSC_ALG1, 
                                &tp_buffer_size)
    )

    CHECK_CUDA( cudaMalloc((void**) &tp_buffer, tp_buffer_size * sizeof(mytype))   )

    CHECK_CUSPARSE( cusparseCsr2cscEx2(handle, M, M, nnz, 
                                dB_csr_values0,     // nnz
                                dB_csr_offsets0,    // M + 1
                                dB_csr_columns0,    // nnz
                                dB_csr_values,      // nnz
                                dB_csr_offsets,     // M + 1
                                dB_csr_columns,     // nnz
                                cudatatype, 
                                CUSPARSE_ACTION_NUMERIC, 
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUSPARSE_CSR2CSC_ALG1, 
                                tp_buffer)
    )

    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&Bdescr, M, M, nnz,
                                      dB_csr_offsets, dB_csr_columns, dB_csr_values, 
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, cudatatype) )

    CHECK_CUDA( cudaFree(dB_csr_columns0) )
    CHECK_CUDA( cudaFree(dB_csr_offsets0) )
    CHECK_CUDA( cudaFree(dB_csr_values0) )

//----------------------------------------------------------------------------
    // prepare C
    // mytype* hC = (mytype *)malloc(sizeof(mytype) * M * M);

    int *dC_csrOffsets, *dC_columns;
    mytype *dC_values;
    // allocate C offsets
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,
                           (M + 1) * sizeof(int)) )
    CHECK_CUSPARSE( cusparseCreateCsr(&Cdescr, M, M, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, cudatatype) )


//------------------------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // Only opA, opB equal to CUSPARSE_OPERATION_NON_TRANSPOSE are supported
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
	cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, Adescr, Bdescr, &beta, Cdescr,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, Adescr, Bdescr, &beta, Cdescr,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, Adescr, Bdescr, &beta, Cdescr,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    // printf("\n***计时***\n");
    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;

    for(int i=0; i < WARMUP+TESTTIME; i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        // compute the intermediate product of A * B
        CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                            &alpha, Adescr, Bdescr, &beta, Cdescr,
                                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc, &bufferSize2, dBuffer2) )


        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i>=WARMUP) sum += elapsed; 
    }
    double cusparsetime=sum/((TESTTIME));
    // printf("cuSPARSE N=%d nnzA=%d(nnz rate=%f) nnzC=%d(nnz rate=%f) \n平均执行时间=%fs\n",N,totalANnz,(double)totalANnz/M/M,nnzC,(double)nnzC/M/M,cusparsetime);
    // printf("cuSPARSE N=%d nnzA=%d(nnz rate=%f) \n平均执行时间=%fs\n",N,totalANnz,(double)totalANnz/M/M,cusparsetime);

    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(Cdescr, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )
    // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )
    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(Cdescr, dC_csrOffsets, dC_columns, dC_values) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, Adescr, Adescr, &beta, Cdescr,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(Adescr) )
    CHECK_CUSPARSE( cusparseDestroySpMat(Bdescr) )
    CHECK_CUSPARSE( cusparseDestroySpMat(Cdescr) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
 
    return cusparsetime;
}


int main(int argc, char **argv) {
    vector<int> sz_vec{1024, 1024, 1024, 8192, 8192, 8192};
    vector<mytype> tr_vec{0.036, 0.038, 0.04, 0.031, 0.033, 0.039};
    ofstream out("cusparse_csr_time.csv");
    for (int i = 0; i < 6; i ++) {
        double cost = test_cusparse_csr(sz_vec[i], sz_vec[i], tr_vec[i]);
        cout << sz_vec[i] << ", " << (float)tr_vec[i] << ": " << cost << endl;
        out << sz_vec[i] << ", " << (float)tr_vec[i] << ", " << cost << endl;
    }
    return 0;
}
