
//#include <cublasXt.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <vector>
#include <tuple>
#include <fstream>
#include <iostream>
#include <omp.h>

// using std::vector;
// using std::tuple;
// using std::endl;
using namespace std;

#define mytype float
#define USINGHALF 0
#define DEVICEDIM 1
#define TESTTIME 10
#define WARMUP 3


mytype get_rand() {
    #if USINGHALF
    return __float2half( ((float)rand()) / (float)(RAND_MAX / 1000) );
    #else
    return ((float)rand()) / (float)(RAND_MAX / 1000);
    #endif
}

void get_random_array(mytype* arr, int len) {
	#pragma omp parallel
    for (int i = 0; i < len; i ++) {
        arr[i] = get_rand();
    }
}

//使用gpu计算矩阵乘
double run_cublas_time(int m, int k, int n){

    mytype* hA;
	mytype* DA;
    mytype* hB;
	mytype* DB;
	hA = (mytype*) malloc(sizeof(mytype) * m * k);
	hB = (mytype*) malloc(sizeof(mytype) * k * n);
    get_random_array(hA, m * k);
    get_random_array(hB, k * n);

    cudaMalloc((void **)&DA, sizeof(mytype)*m*k);
    cudaMalloc((void **)&DB, sizeof(mytype)*k*n);
    cudaMemcpy(DA, hA, m * k * sizeof (mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(DB, hB, k * n * sizeof (mytype), cudaMemcpyHostToDevice);



    const mytype alf = 1.0f;
    const mytype bet = 0.0f;
    const mytype *alpha = &alf;
    const mytype *beta = &bet;
    
    mytype* C;

    cudaMalloc((void **)&C, sizeof(mytype)*m*n);
    
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    // cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH);

    // cublasStatus_t status;
    // cublasXtHandle_t handle;
    
    //int devices[DEVICEDIM];
    //int num_of_devices = DEVICEDIM;
    //for(int i=0;i<DEVICEDIM;i++){
    //    devices[i]=i;
    //}

    
    //cudaDeviceProp deviceProp;
    // printf("Using %d GPUs\n", num_of_devices);
    //for (int i = 0; i < num_of_devices; i++) {
    //    cudaGetDeviceProperties(&deviceProp, devices[i]);
        // printf("GPU ID = %d, Name = %s \n", devices[i], deviceProp.name);
    //}

    // status = cublasXtCreate(&handle);
    // status = cublasXtDeviceSelect(handle,num_of_devices, devices);
    // mytype *AA=copy_B2(DA,m, k);
    // mytype *BB=copy_B2(DB,k, n);

    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;

    for(int i=0; i < 10 + WARMUP; i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);


        #if !USINGHALF
        // cublasXtSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, AA, k, BB, n, beta, C, m);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, DA, k, DB, n, beta, C, m); 
        #else
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, DA, k, DB, n, beta, C, m); 
        #endif
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i>=WARMUP) sum += elapsed;
    }
    // printf("cuBLAS平均执行时间=%fs\n",sum/(TESTTIME-WARMUP));
	cudaFree(DA);
	cudaFree(DB);
	cudaFree(C);
	free(hA);
	free(hB);
    return (double)(sum/(TESTTIME));
}

int main() {
    // int m, k, n;
    vector<tuple<int, int, int> > problem_size = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        {16384, 16384, 16384},
        {32768, 32768, 32768},
        {13656, 13656, 13656},
        {128, 576, 25600},
        {256, 1152, 6400}
    };
    
    ofstream out("cublas_time.csv");
    for (auto& [m, k, n] : problem_size) {
        cout << "running " << m << "*" << k << "*" << n << endl;
        double cost = run_cublas_time(m, k, n);
        out << m << ", " << k << ", " << n << ", " << cost << endl;
    }
    out.close();
    return 0;
}
