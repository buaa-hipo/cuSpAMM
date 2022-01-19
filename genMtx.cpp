#include <sys/time.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
using namespace std;

//代数衰减 |a_ij| < c/(|i-j|^v+1); c>0, v>0
// 增加了截断
int getDecayMatrixAlg(float* A, float c, float v,int m,int n, float thresh){
    float t;
    int nnz = 0;
    #pragma omp parallel
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            t = (float) c / ((float) pow(abs(i-j),v)+1);
            // A[i*n+j] = t;
            if (t < thresh) A[i*n+j] = 0;
            else {
                A[i*n+j] = t;//((mytype)(rand()%10))/10*t;
                nnz ++;
            }
            // if(rand()%2==0) A[i*n+j]*=-1;
        }
    }
    return nnz;
}


void gen_mtx(float* A, int m, int n, int nnz, string filename){
    float t;
    ofstream mtxfile(filename);
    mtxfile << "%%MatrixMarket matrix coordinate real general" << endl;
    mtxfile << m << " " << n << " " << nnz << endl;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if (A[i*n+j] != 0) {
                mtxfile << i << " " << j << " " << A[i*n+j] << endl;
            }
        }
    }
    mtxfile.close();
}

int main() {
    vector<int> sz_vec{1024, 1024, 1024, 8192, 8192, 8192};
    vector<float> tr_vec{0.036, 0.038, 0.04, 0.031, 0.033, 0.039};
    float* small = (float*) malloc(sizeof(float) * 1024 * 1024);
    float* big = (float*) malloc(sizeof(float) * 8192 * 8192);
    int i = 1;
    for (auto tr : tr_vec) {
        int nnz = getDecayMatrixAlg(small, 0.1, 0.1, 1024, 1024, tr);
        gen_mtx(small, 1024, 1024, nnz, "decay1024_" + to_string(i++) + ".mtx");
    }
    i = 1;
    for (auto tr : tr_vec) {
        int nnz = getDecayMatrixAlg(big, 0.1, 0.1, 8192, 8192, tr);
        gen_mtx(big, 8192, 8192, nnz, "decay8192_" + to_string(i++) + ".mtx");
    }
    return 0;
}
