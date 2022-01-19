//#include "main.h"
//#include <cublasXt.h>
#include <cuda_runtime.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <typeinfo>
extern float Norm;

#define COUNTERRTRANS(A,B,m,n) \
float ef=0;\
float cf=0;\
float c_f=0;\
for(int i=0;i<n;i++){ \
    for(int j=0;j<m;j++){ \
        float t;\
        if(typeid(A[0])!=typeid(t)){\
            t=__half2float(A[i*m+j]);\
        }\
        else{\
            t=A[i*m+j];\
        }\
        ef+=fabs(t-B[i*m+j])*fabs(t-B[i*m+j]);\
        cf+=fabs(t)*fabs(t);\
        c_f+=fabs(B[i*m+j])*fabs(B[i*m+j]);\
    } \
} \
printf("误差矩阵范数=%f, 稠密矩阵乘结果范数=%f, SpAMM结果范数=%f\n",sqrt(ef),sqrt(cf),sqrt(c_f));

#define CHECKEQ6(A,B,m,n) \
float ef6=0;\
float cf6=0;\
float c_f6=0;\
for(int i=0;i<m;i++){ \
    for(int j=0;j<n;j++){ \
        float t;\
        if(typeid(A[0])!=typeid(t)){\
            t=__half2float(A[i*n+j]);\
        }\
        else{\
            t=A[i*n+j];\
        }\
        ef6+=fabs(t-B[i*n+j])*fabs(t-B[i*n+j]);\
        cf6+=fabs(t)*fabs(t);\
        c_f6+=fabs(B[i*n+j])*fabs(B[i*n+j]);\
    } \
} \
printf("EF=%f CF=%f C'F=%f\n",sqrt(ef6),sqrt(cf6),sqrt(c_f6));

#define COUNTERR(A,B,m,n) \
float err=0;\
float sum=0;\
for(int i=0;i<m;i++){ \
    for(int j=0;j<n;j++){ \
        err+=fabs(A[i*n+j]-B[i*n+j]);\
        sum+=fabs(A[i*n+j]);\
    } \
} \
printf("ERRSUM=%f AVR=%f sumA=%f\n",err,err/(m*n),sum/(m*n));

#define CHECKEQ(A,B,m,n) \
for(int i=0;i<m;i++){ \
    for(int j=0;j<n;j++){ \
        if(fabs(A[i*n+j]-B[i*n+j])>=E){ \
            printf("ERROR! (%d,%d) cpu=%f gpu=%f\n",i,j,A[i*n+j],B[i*n+j]); \
        } \
    } \
} \
printf("CHECK DONE\n");


mytype* C;

//!判断6位有效数字，是有效数字，不是小数点后的位数
float DECIMAL(float n){
    while(1){
        if(fabs(n)<1) return n;
        else n/=10;
    }
}


int getMatrixFromMTX(mytype* A, int m, int n, std::string filename, float thresh){
    std::cout << "start reading matrix" << std::endl;
    //从mtx文件读入矩阵
    std::ifstream finA(filename);
    std::string line;
    // int i=0;
    int nnz = 0;
    while (getline(finA, line)){
        // if(i==0){
        //     i++;
        //     continue;
        // }
        // nnz ++;
        
        std::istringstream sin(line);
        std::vector<std::string> Waypoints;
        std::string info;
        std::stringstream input(line);
        std::stringstream srow;
        std::stringstream scol;
        std::stringstream sval;
        std::string x_str;
        int row, col;
        double val;

        while(input>>x_str){
            Waypoints.push_back(x_str);
        }

        srow << Waypoints[0];
        srow >> row;
        scol << Waypoints[1];
        scol >> col;
        sval << Waypoints[2];
        sval >> val;
        // if(row>=m) printf("%d %d %d %d\n",row,col,m,n);
        
        if (fabs(val) > thresh) {
            nnz ++;

            #if !USINGHALF
            A[(row-1)*n+(col-1)] = val;
            #else
            A[(row-1)*n+(col-1)] = __float2half(val);
            #endif
        }

    }
    std::cout << "finished reading matrix" << std::endl;
    return nnz;
    // MATRIXSHOW21D(A,M,K);
    // MATRIXSHOW21D(B,K,N);
}




//转置B矩阵
float* copy_B(mytype* B,int m,int n){
    float *b;
    cudaMallocManaged((void **)&b, sizeof(float)*m*n);
    // for(int i=0;i<n;i++){
    //     for(int j=0;j<m;j++){
    //         #if USINGHALF
    //         b[i*m+j]=__half2float(B[i*m+j]);
    //         #else
    //         b[i*m+j]=B[i*m+j];
    //         #endif
    //     }
    // }
    // MATRIXSHOW21D(b,m,n);
    
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            #if USINGHALF
            b[i*n+j]=__half2float(B[i*n+j]);
            #else
            b[i*n+j]=B[i*n+j];
            #endif
        }
    }
    
    return b;

    
    // MATRIXSHOW21D(B,M,N);
}

//转置B矩阵
mytype* copy_B2(mytype* B,int m,int n){
    mytype *b;
    cudaMallocManaged((void **)&b, sizeof(mytype)*m*n);
    
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            #if USINGHALF
            b[i*n+j]=__half2float(B[i*n+j]);
            #else
            b[i*n+j]=B[i*n+j];
            #endif
        }
    }
    
    return b;

    
    // MATRIXSHOW21D(B,M,N);
}
