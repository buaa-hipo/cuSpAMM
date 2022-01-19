
#include <cublasXt.h>
#include <cuda_runtime.h>
#include <cuda.h>
// #include <cublas_v2.h>
#include <vector>
#include <tuple>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <cuda_fp16.h>
#include <omp.h>

// using std::vector;
// using std::tuple;
// using std::endl;
using namespace std;

#define mytype float
#define USINGHALF 0
#define TESTTIME 10
#define WARMUP 3

int DEVICEDIM = 1;

mytype get_rand() {
    return ((float)rand()) / (float)(RAND_MAX / 1000);
}

void get_random_array(mytype* arr, int len) {
    #pragma omp parallel
    for (int i = 0; i < len; i ++) {
        arr[i] = get_rand();
    }
}


//使用gpu计算矩阵乘
double run_cublasxt_time(int m, int k, int n){

    // get_random_array(HA, n * n);
    // get_random_array(HB, n * n);
    mytype* DA;
    mytype* DB;

    mytype* hA = (mytype*) malloc(m * k * sizeof (mytype));
    mytype* hB = (mytype*) malloc(k * n * sizeof (mytype));
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

    cublasStatus_t status;
    cublasXtHandle_t handle;
    
    int devices[DEVICEDIM];
    int num_of_devices = 0;
    cudaGetDeviceCount(&num_of_devices);
    assert (num_of_devices >= DEVICEDIM);

    for(int i=0;i<DEVICEDIM;i++){
        devices[i]=i;
    }
    num_of_devices = DEVICEDIM;
    
    cudaDeviceProp deviceProp;
    // printf("Using %d GPUs\n", num_of_devices);
    for (int i = 0; i < num_of_devices; i++) {
        cudaGetDeviceProperties(&deviceProp, devices[i]);
        printf("GPU ID = %d, Name = %s \n", devices[i], deviceProp.name);
    }

    status = cublasXtCreate(&handle);
    assert (status == CUBLAS_STATUS_SUCCESS);
    status = cublasXtDeviceSelect(handle, num_of_devices, devices);
    assert (status == CUBLAS_STATUS_SUCCESS);


    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;
    cout << "start computing" << endl;
    for(int i=0; i < TESTTIME + WARMUP; i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);


        #if !USINGHALF
        status = cublasXtSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, DA,
                         k, DB, n, beta, C, m);
        // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, DA, k, DB, n, beta, C, m); 
        #else
        status = cublasXtHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, DA,
                         k, DB, n, beta, C, m);
        // cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, AA, k, BB, n, beta, C, m); 
        #endif
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i>=WARMUP) sum += elapsed;
    }
    // printf("cuBLAS平均执行时间=%fs\n",sum/(TESTTIME-WARMUP));

    free(hA);
    free(hB);
    cudaFree(DA);
    cudaFree(DB);
    cudaFree(C);
    cublasXtDestroy(handle);

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
    
    ofstream out("cublasxt_time.csv");
    out << "problem size, ";
    for (auto& [m, k, n] : problem_size) {
        out << m << "*" << k << "*" << n << ", ";
    }
    out << endl;
    for (int i = 1; i <= 2; i *= 2) {
        out << i << ", ";
        DEVICEDIM = i; // number of devices
        for (auto& [m, k, n] : problem_size) {
            cout << "running " << m << "*" << k << "*" << n << " on " << i << " GPUs." << endl;
            double cost = run_cublasxt_time(m, k, n);
            out << cost << ", ";
        }
        out << endl;
    }
    out.close();
    return 0;
}
