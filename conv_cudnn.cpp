#include <iostream>
#include <cassert>
#include <sys/time.h>
#include <string>

#include <cuda_runtime.h>
#include <cudnn.h>

// #include "help.h"
// #include "im2col.h"
// #include "conv.h"

void cudnnConvolutionForwardFP32(
    float              *data_input,     float              *data_filter,
    float              *data_output,    const int           batch_size,     
    const int           input_c,        const int           input_h,
    const int           input_w,        const int           filter_n,
    const int           filter_h,       const int           filter_w,
    const int           output_h,       const int           output_w,
    const int           padding_h,      const int           padding_w,
    const int           stride_h,       const int           stride_w,        bool                ifTensorCore)
{
    float costTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int input_size  = batch_size * input_c * input_h * input_w, 
        filter_size = filter_n * input_c * filter_h * filter_w,
        output_size = batch_size * filter_n * output_h * output_w;
    
    // copy to gpu memory
    float   *d_input    = NULL,
            *d_filter   = NULL;
    float   *d_output   = NULL;    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMemcpy(d_input, data_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_filter, filter_size * sizeof(float));
    cudaMemcpy(d_filter, data_filter, filter_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMemset(d_output, 0, output_size * sizeof(float));
 
    // cudnn variables
    cudnnHandle_t cudnn;
    cudnnTensorFormat_t inout_layout = (layout == NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    
    cudnnCreate(&cudnn);
    // input descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                    inout_layout,
                                    CUDNN_DATA_FLOAT,
                                    batch_size,
                                    input_c,
                                    input_h,
                                    input_w));
    // filter descriptor 
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        filter_n,
                                        input_c,
                                        filter_h,
                                        filter_w));
    // convolution descriptor 
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                            padding_h,
                                            padding_w,
                                            stride_h,
                                            stride_w,
                                            /*dilation_height=*/1,
                                            /*dilation_width=*/1,
                                            CUDNN_CROSS_CORRELATION,
                                            CUDNN_DATA_FLOAT));
    // output descriptor 
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        inout_layout,
                                        CUDNN_DATA_FLOAT,
                                        batch_size,
                                        filter_n,
                                        output_h,
                                        output_w));

    // set mathtype and algorithm
    if (ifTensorCore) {     // with tensor core
        checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor,
                                        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
        convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else {                // without tensor core
        checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor,
                                        CUDNN_DEFAULT_MATH));
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        filter_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));                                
    }

    // get size of workspace
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                    input_descriptor,
                                                    filter_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    convolution_algorithm,
                                                    &workspace_bytes));
    void *d_workspace = NULL;
    cudaMalloc(&d_workspace, workspace_bytes);

    // warmup
    for (int i = 0; i < 10; i++) {
        checkCUDNN(cudnnConvolutionForward(cudnn,
                                        &alpha,
                                        input_descriptor,
                                        d_input,
                                        filter_descriptor,
                                        d_filter,
                                        convolution_descriptor,
                                        convolution_algorithm,
                                        d_workspace,
                                        workspace_bytes,
                                        &beta,
                                        output_descriptor,
                                        d_output));
        cudaDeviceSynchronize();
    }
    
    cudaEventRecord(start, 0);
    for (int i = 0; i < TIMES; i++) {
        checkCUDNN(cudnnConvolutionForward(cudnn,
                                        &alpha,
                                        input_descriptor,
                                        d_input,
                                        filter_descriptor,
                                        d_filter,
                                        convolution_descriptor,
                                        convolution_algorithm,
                                        d_workspace,
                                        workspace_bytes,
                                        &beta,
                                        output_descriptor,
                                        d_output));
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&costTime, start, stop);
    std::cout << costTime/TIMES << " ms" << std::endl;

    // copy back
    cudaMemcpy(data_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_workspace);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudnnDestroy(cudnn);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    #define CUDA_MEM
    #ifdef  CUDA_MEM
    int cudaMemBytes = (input_size+filter_size+output_size)*sizeof(float)
                    + workspace_bytes;
    std::cout << "GPU MEM:\t\t\t" << cudaMemBytes/1024/1024 << "MB" << std::endl; 
    #endif
}

// void cudnnConvolutionForwardFP16(
//     half               *data_input,     half               *data_filter,
//     half               *data_output,    const int           batch_size,     
//     const int           input_c,        const int           input_h,
//     const int           input_w,        const int           filter_n,
//     const int           filter_h,       const int           filter_w,
//     const int           output_h,       const int           output_w,
//     const int           padding_h,      const int           padding_w,
//     const int           stride_h,       const int           stride_w,
//     TensorLayout        layout,         bool                ifTensorCore)
// {
//     float costTime;
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     int input_size  = batch_size * input_c * input_h * input_w, 
//         filter_size = filter_n * input_c * filter_h * filter_w,
//         output_size = batch_size * filter_n * output_h * output_w;

//     // copy to gpu memory
//     half    *d_input    = NULL,
//             *d_filter   = NULL,
//             *d_output   = NULL;    
//     cudaMalloc(&d_input, input_size * sizeof(half));
//     cudaMemcpy(d_input, data_input, input_size * sizeof(half), cudaMemcpyHostToDevice);
//     cudaMalloc(&d_filter, filter_size * sizeof(half));
//     cudaMemcpy(d_filter, data_filter, filter_size * sizeof(half), cudaMemcpyHostToDevice);
//     cudaMalloc(&d_output, output_size * sizeof(half));
//     cudaMemset(d_output, 0, output_size * sizeof(half));
 
//     // cudnn variables
//     cudnnHandle_t cudnn;
//     cudnnTensorFormat_t inout_layout = (layout == NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
//     cudnnTensorDescriptor_t input_descriptor, output_descriptor;
//     cudnnFilterDescriptor_t filter_descriptor;
//     cudnnConvolutionDescriptor_t convolution_descriptor;
//     cudnnConvolutionFwdAlgo_t convolution_algorithm;
    
//     cudnnCreate(&cudnn);
//     // input descriptor
//     checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
//     checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
//                                     inout_layout,
//                                     CUDNN_DATA_HALF,
//                                     batch_size,
//                                     input_c,
//                                     input_h,
//                                     input_w));
//     // filter descriptor 
//     checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
//     checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
//                                         CUDNN_DATA_HALF,
//                                         CUDNN_TENSOR_NCHW,
//                                         filter_n,
//                                         input_c,
//                                         filter_h,
//                                         filter_w));
//     // convolution descriptor 
//     checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
//     checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
//                                             padding_h,
//                                             padding_w,
//                                             stride_h,
//                                             stride_w,
//                                             /*dilation_height=*/1,
//                                             /*dilation_width=*/1,
//                                             CUDNN_CROSS_CORRELATION,
//                                             CUDNN_DATA_HALF));
//     // output descriptor 
//     checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
//     checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
//                                         inout_layout,
//                                         CUDNN_DATA_HALF,
//                                         batch_size,
//                                         filter_n,
//                                         output_h,
//                                         output_w));

//     // set mathtype
//     if (ifTensorCore) {     // with tensor core
//         checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor,
//                                         CUDNN_TENSOR_OP_MATH));
//     } else {                // without tensor core
//         checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor,
//                                         CUDNN_DEFAULT_MATH));                                
//     }
//     // get algorithm
//     checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
//                                         input_descriptor,
//                                         filter_descriptor,
//                                         convolution_descriptor,
//                                         output_descriptor,
//                                         CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//                                         /*memoryLimitInBytes=*/0,
//                                         &convolution_algorithm));

//     // get size of workspace
//     size_t workspace_bytes = 0;
//     checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
//                                                     input_descriptor,
//                                                     filter_descriptor,
//                                                     convolution_descriptor,
//                                                     output_descriptor,
//                                                     convolution_algorithm,
//                                                     &workspace_bytes));
//     void *d_workspace = NULL;
//     cudaMalloc(&d_workspace, workspace_bytes);

//     // warmup
//     for (int i = 0; i < 10; i++) {
//         checkCUDNN(cudnnConvolutionForward(cudnn,
//                                         &alpha,
//                                         input_descriptor,
//                                         d_input,
//                                         filter_descriptor,
//                                         d_filter,
//                                         convolution_descriptor,
//                                         convolution_algorithm,
//                                         d_workspace,
//                                         workspace_bytes,
//                                         &beta,
//                                         output_descriptor,
//                                         d_output));
//         cudaDeviceSynchronize();                                
//     }
    
//     cudaEventRecord(start, 0);
//     for (int i = 0; i < TIMES; i++) {
//         checkCUDNN(cudnnConvolutionForward(cudnn,
//                                         &alpha,
//                                         input_descriptor,
//                                         d_input,
//                                         filter_descriptor,
//                                         d_filter,
//                                         convolution_descriptor,
//                                         convolution_algorithm,
//                                         d_workspace,
//                                         workspace_bytes,
//                                         &beta,
//                                         output_descriptor,
//                                         d_output));
//         cudaDeviceSynchronize();                                
//     }
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&costTime, start, stop);
//     std::cout << costTime/TIMES << " ms" << std::endl;

//     // copy back
//     cudaMemcpy(data_output, d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost);

//     cudaFree(d_workspace);
//     cudaFree(d_input);
//     cudaFree(d_filter);
//     cudaFree(d_output);
//     cudnnDestroy(cudnn);
//     cudnnDestroyTensorDescriptor(input_descriptor);
//     cudnnDestroyFilterDescriptor(filter_descriptor);
//     cudnnDestroyTensorDescriptor(output_descriptor);
//     cudnnDestroyConvolutionDescriptor(convolution_descriptor);
//     #define CUDA_MEM
//     #ifdef  CUDA_MEM
//     int cudaMemBytes = (input_size+filter_size+output_size)*sizeof(half)
//                     + workspace_bytes;
//     std::cout << "GPU MEM:\t\t\t" << cudaMemBytes/1024/1024 << "MB" << std::endl; 
//     #endif
// }

// void cudnnConvolutionForwardINT8(
//     unsigned char      *data_input,     unsigned char      *data_filter,
//     unsigned char      *data_output,    const int           batch_size,     
//     const int           input_c,        const int           input_h,
//     const int           input_w,        const int           filter_n,
//     const int           filter_h,       const int           filter_w,
//     const int           output_h,       const int           output_w,
//     const int           padding_h,      const int           padding_w,
//     const int           stride_h,       const int           stride_w,
//     TensorLayout        layout)
// {
//     float costTime;
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     int input_size  = batch_size * input_c * input_h * input_w, 
//         filter_size = filter_n * input_c * filter_h * filter_w,
//         output_size = batch_size * filter_n * output_h * output_w;
    
//     // copy to gpu memory
//     unsigned char   *d_input    = NULL,
//                     *d_filter   = NULL,
//                     *d_output   = NULL;    
//     cudaMalloc(&d_input, input_size * sizeof(unsigned char));
//     cudaMemcpy(d_input, data_input, input_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
//     cudaMalloc(&d_filter, filter_size * sizeof(unsigned char));
//     cudaMemcpy(d_filter, data_filter, filter_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
//     cudaMalloc(&d_output, output_size * sizeof(unsigned char));
//     cudaMemset(d_output, 0, output_size * sizeof(unsigned char));
 
//     // cudnn variables
//     cudnnHandle_t cudnn;
//     cudnnTensorFormat_t inout_layout = (layout == NCHW) ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
//     cudnnTensorDescriptor_t input_descriptor, output_descriptor;
//     cudnnFilterDescriptor_t filter_descriptor;
//     cudnnConvolutionDescriptor_t convolution_descriptor;
//     cudnnConvolutionFwdAlgo_t convolution_algorithm;
    
//     cudnnCreate(&cudnn);
//     // input descriptor
//     checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
//     checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
//                                     inout_layout,
//                                     CUDNN_DATA_INT8,
//                                     batch_size,
//                                     input_c,
//                                     input_h,
//                                     input_w));
//     // filter descriptor 
//     checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
//     checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
//                                         CUDNN_DATA_INT8,
//                                         CUDNN_TENSOR_NHWC,
//                                         filter_n,
//                                         input_c,
//                                         filter_h,
//                                         filter_w));
//     // convolution descriptor 
//     checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
//     checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
//                                             padding_h,
//                                             padding_w,
//                                             stride_h,
//                                             stride_w,
//                                             /*dilation_height=*/1,
//                                             /*dilation_width=*/1,
//                                             CUDNN_CROSS_CORRELATION,
//                                             CUDNN_DATA_INT32));
//     // output descriptor 
//     checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
//     checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
//                                         inout_layout,
//                                         CUDNN_DATA_INT8,
//                                         batch_size,
//                                         filter_n,
//                                         output_h,
//                                         output_w));

//     // set mathtype and algorithm
//     checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor,
//                                     CUDNN_DEFAULT_MATH));
//     // checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
//     //                                 input_descriptor,
//     //                                 filter_descriptor,
//     //                                 convolution_descriptor,
//     //                                 output_descriptor,
//     //                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//     //                                 /*memoryLimitInBytes=*/0,
//     //                                 &convolution_algorithm));           

//     convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;                     

//     // get size of workspace
//     size_t workspace_bytes = 0;
//     checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
//                                                     input_descriptor,
//                                                     filter_descriptor,
//                                                     convolution_descriptor,
//                                                     output_descriptor,
//                                                     convolution_algorithm,
//                                                     &workspace_bytes));
//     void *d_workspace = NULL;
//     cudaMalloc(&d_workspace, workspace_bytes);

//     // warmup
//     for (int i = 0; i < 10; i++) {
//         checkCUDNN(cudnnConvolutionForward(cudnn,
//                                         &alpha,
//                                         input_descriptor,
//                                         d_input,
//                                         filter_descriptor,
//                                         d_filter,
//                                         convolution_descriptor,
//                                         convolution_algorithm,
//                                         d_workspace,
//                                         workspace_bytes,
//                                         &beta,
//                                         output_descriptor,
//                                         d_output));
//         cudaDeviceSynchronize();
//     }
    
//     cudaEventRecord(start, 0);
//     for (int i = 0; i < TIMES; i++) {
//         checkCUDNN(cudnnConvolutionForward(cudnn,
//                                         &alpha,
//                                         input_descriptor,
//                                         d_input,
//                                         filter_descriptor,
//                                         d_filter,
//                                         convolution_descriptor,
//                                         convolution_algorithm,
//                                         d_workspace,
//                                         workspace_bytes,
//                                         &beta,
//                                         output_descriptor,
//                                         d_output));
//         cudaDeviceSynchronize();
//     }
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&costTime, start, stop);
//     std::cout << costTime/TIMES << " ms" << std::endl;

//     // copy back
//     cudaMemcpy(data_output, d_output, output_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

//     cudaFree(d_workspace);
//     cudaFree(d_input);
//     cudaFree(d_filter);
//     cudaFree(d_output);
//     cudnnDestroy(cudnn);
//     cudnnDestroyTensorDescriptor(input_descriptor);
//     cudnnDestroyFilterDescriptor(filter_descriptor);
//     cudnnDestroyTensorDescriptor(output_descriptor);
//     cudnnDestroyConvolutionDescriptor(convolution_descriptor);
//     #define CUDA_MEM
//     #ifdef  CUDA_MEM
//     int cudaMemBytes = (input_size+filter_size+output_size)*sizeof(unsigned char)
//                     + workspace_bytes;
//     std::cout << "GPU MEM:\t\t\t" << cudaMemBytes/1024/1024 << "MB" << std::endl; 
//     #endif
// }