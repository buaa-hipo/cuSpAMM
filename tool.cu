#include "main.h"
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
        ef+=fabs(t-B[j*n+i])*fabs(t-B[j*n+i]);\
        cf+=fabs(t)*fabs(t);\
        c_f+=fabs(B[j*n+i])*fabs(B[j*n+i]);\
    } \
} \
printf("EF=%f CF=%f C'F=%f\n",sqrt(ef),sqrt(cf),sqrt(c_f));

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

//衰减矩阵乘
void check(float* A,float* B,float* in_C){
    // cudaMallocHost((void **)&C, sizeof(float)*T*T);
    // printf("***CPU SpAMM checking***\n");
    // int count=0;
    
    // //直接变成非递归 小块乘
    // for(int Ci=0;Ci<M/LoNum;Ci++){
    //     for(int Cj=0;Cj<N/LoNum;Cj++){
    //         //计算小块C[Ci,Cj]的结果
    //         //遍历A,B小块
    //         for(int Ki=0;Ki<K/LoNum;Ki++){
    //             //计算小块范数A[Ci,Ki]，B[Ki,Cj]
    //             float normA=0.0f,normB=0.0f,norm_mul;
    //             //遍历块内元素
    //             for(int i=0;i<LoNum;i++){
    //                 for(int j=0;j<LoNum;j++){
    //                     normA += A[(Ci*LoNum+i)*T+Ki*LoNum+j] * A[(Ci*LoNum+i)*T+Ki*LoNum+j];
    //                 }
    //             }
    //             for(int i=0;i<LoNum;i++){
    //                 for(int j=0;j<LoNum;j++){
    //                     normB += B[(Ki*LoNum+i)*T+Cj*LoNum+j] * B[(Ki*LoNum+i)*T+Cj*LoNum+j];
    //                 }
    //             }
    //             norm_mul = sqrt(normA)*sqrt(normB);
    //             // printf("norm=%f A=%f B=%f\n",norm,sqrt(normA),sqrt(normB));
    //             if(norm_mul>Norm){
    //                 count++;
    //                 //计算小块乘C[Ci,Cj] = A[Ci,Ki] * B[Ki,Cj]
    //                 for(int i=0;i<LoNum;i++){
    //                     for(int j=0;j<LoNum;j++){
    //                         for(int k=0;k<LoNum;k++){
    //                             C[(Ci*LoNum+i)*T+Cj*LoNum+j] += A[(Ci*LoNum+i)*T+Ki*LoNum+k] * B[(Ki*LoNum+k)*T+Cj*LoNum+j];
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // // MATRIXSHOW2D(C,M,N);
    // // MATRIXSHOW2D(gpu_C,M,N);
    // COUNTERR(C,in_C,M,N);
    // CHECKEQ6(C,in_C,M,N);
    
    // // printf("valid mul=%d, simple mul=%d, rate=%f\n",count,(M/LoNum)*(N/LoNum)*(K/LoNum),(float)count/((M/LoNum)*(N/LoNum)*(K/LoNum)));
}

//判断矩阵乘，非SpAMM算法
void check_simple_matrix_mul(float* A,float* B,float* in_C){
    // cudaMallocHost((void **)&C, sizeof(float)*T*T);
    // printf("***CPU ERROR checking***\n");
    
    // //矩阵乘
    // for(int i=0;i<M;i++){
    //     for(int j=0;j<N;j++){
    //         for(int k=0;k<K;k++){
    //             C[i*N+j] += A[i*N+k] * B[k*N+j];
    //         }
    //     }
    // }

    // // MATRIXSHOW2D(C,M,N);
    // MATRIXSHOW21D(in_C,M,N);

    // // CHECKEQ6(C,gpu_C,M,N);
    // COUNTERR(C,in_C,M,N);

}

//使用gpu计算矩阵乘
void run_cublas_time(mytype* A,mytype* B){

    int lda, ldb, ldc, m, n, k;
    const mytype alf = 1.0f;
    const mytype bet = 0.0f;
    const mytype *alpha = &alf;
    const mytype *beta = &bet;
    m=M;
    n=N;
    k=K;
	lda = m;
	ldb = k;
	ldc = n;

    cudaMallocHost((void **)&C, sizeof(float)*M*N);
    
    //矩阵乘
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    mytype *AA=copy_B(A,M,K);
    mytype *BB=copy_B(B,K,N);

    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum=0.0;

    for(int i=0;i<TESTTIME;i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        #if !USINGHALF
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, AA, k, BB, n, beta, C, m); 
        #else
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, AA, k, BB, n, beta, C, m); 
        #endif
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        if(i>=WARMUP) sum += elapsed; //测速的时候改成3
    }
    printf("cublas time=%fs\n",sum/(TESTTIME-WARMUP));

}

//使用gpu计算矩阵乘
void check_simple_gpu(mytype* A,mytype* B,float* in_C){

    int lda, ldb, ldc, m, n, k;
    const mytype alf = 1.0f;
    const mytype bet = 0.0f;
    const mytype *alpha = &alf;
    const mytype *beta = &bet;
    m=M;
    n=N;
    k=K;
	lda = m;
	ldb = k;
	ldc = n;

    cudaMallocHost((void **)&C, sizeof(float)*M*N);
    printf("***GPU ERROR checking***\n");
    
    //矩阵乘
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    mytype *AA=copy_B(A,M,K);
    mytype *BB=copy_B(B,K,N);
    #if !USINGHALF
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, AA, k, BB, n, beta, C, m); 
    #else
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, AA, k, BB, n, beta, C, m); 
    #endif

    cudaDeviceSynchronize();

    // float* hand_C;
    // cudaMallocHost((void **)&hand_C, sizeof(float)*M*N);
    // for(int i=0;i<M;i++){
    //     for(int j=0;j<N;j++){
    //         hand_C[i*N+j] = 0;
    //         for(int k=0;k<K;k++){
    //             hand_C[i*N+j]+=A[i*K+k]*B[k*N+j];
    //         }
    //     }
    // }
    // CHECKEQ(hand_C,in_C,M,N);
    COUNTERRTRANS(C,in_C,M,N);

    // mytype *TC=trans_B(C);
    // MATRIXSHOW21D(TC,M,N);
    
}

void getMatrixFromMTX(mytype* A,int m,int n,std::string filename){
    //从csv文件读入矩阵
    std::ifstream finA(filename);
    std::string line;
    int i=0;
    while (getline(finA, line)){
        if(i==0){
            i++;
            continue;
        } 
        
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
        
        #if !USINGHALF
        A[row*n+col] = val;
        #else
        A[row*n+col] = __float2half(val);
        #endif

        i++;
    }
    
    // MATRIXSHOW21D(A,M,K);
    // MATRIXSHOW21D(B,K,N);
}

void getMatrixFromCSV(mytype* A,int m,int n,std::string filename){
    //从csv文件读入矩阵
    std::ifstream finA(filename);
    std::string line;
    int i=0;
    while (getline(finA, line)){
        // std::cout << "原始字符串: " << line << std::endl; 
        std::istringstream sin(line);
        std::vector<std::string> Waypoints;
        std::string info;

        while (getline(sin, info, ',')) {
            Waypoints.push_back(info);
        }
        for(int j=0;j<Waypoints.size();j++){
            std::stringstream sx;
            std::string x_str;
            double x;
            x_str = Waypoints[j];
            sx << x_str;
            sx >> x;
            #if !USINGHALF
            A[i*K+j] = x;
            #else
            A[i*K+j] = __float2half(x);
            #endif
        }
        i++;
    }
    // MATRIXSHOW21D(A,M,K);
}


//生成普通的矩阵
void getNormMatrix(mytype* A,mytype* B){
    //初始化矩阵
    for(int i=0;i<inM;i++){
        for(int j=0;j<inK;j++){
            float t;
            t=1;
            // if(j%2==0) t=1;
            if(rand()%3==0) t+=rand()%3;
            // t = (float) (rand()%2);//(float) (rand()%100);
            // t = (float) (rand()%100);
            #if !USINGHALF
            A[i*K+j] = t;
            #else
            A[i*K+j] = __float2half(t);
            #endif
        }
    }
    for(int i=0;i<inK;i++){
        for(int j=0;j<inN;j++){
            float t;
            t=1;
            if(j%2==0) t+=rand()%3;
            // if(i==j){
            //     if(j>=K/2) t=0;
            // }
            // t = (float) (rand()%2);//(float) (rand()%100);
            // t = (float) (rand()%100);
            #if !USINGHALF
            B[i*N+j] = t;
            #else
            B[i*N+j] = __float2half(t);
            #endif
        }
    }
    // B[1]=100;
    // MATRIXSHOW21D(A,M,K);
}

//指数衰减 |a_ij| < c*(v)^|i-j| ; c>0, 1>v>0
void getDecayMatrixExp(mytype* A,float c,float v,int m,int n){
    float t;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            t = c * (float) pow(v,abs(i-j));
            // t = ((float)(rand()%10))/10*t;
            #if !USINGHALF
            A[i*n+j] = t;
            #else
            A[i*n+j] = __float2half(t);
            #endif
        }
    }
}


//代数衰减 |a_ij| < c/(|i-j|^v+1); c>0, v>0
void getDecayMatrixAlg(mytype* A,float c,float v,int m,int n){
    float t;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            t = (float) c / ((float) pow(abs(i-j),v)+1);
            // A[i*n+j] = t;
            A[i*n+j] = t;//((float)(rand()%10))/10*t;
            // if(rand()%2==0) A[i*n+j]*=-1;
        }
    }
}

float tuneValidRate(float *A_normmap, float *B_normmap,int m,int n){
    // printf("*** tuning valid, expected=%f ***\n",ExpectedRate);
    int count=0;
    double norm,ave=0,validRate=0,up=0,down=0,Norm=0;
    int totalNum=(M/LoNum)*(N/LoNum)*(K/LoNum);
    int validC[M/LoNum][N/LoNum];
    //直接变成非递归 小块乘
    for(int Ti=0;Ti<TUNINGTIME;Ti++){
        count=0;
        ave=0;
        for(int Ci=0;Ci<M/LoNum;Ci++){
            for(int Cj=0;Cj<N/LoNum;Cj++){
                //计算小块C[Ci,Cj]的结果
                //遍历A,B小块
                validC[Ci][Cj]=0;
                for(int Ki=0;Ki<K/LoNum;Ki++){
                    //计算范数乘 小块A[Ci,Ki]和B[Ki,Cj]
                    norm = GETELEMENT21(A_normmap,Ci,Ki,K/LoNum)*GETELEMENT21(B_normmap,Ki,Cj,N/LoNum);
                    ave+=norm;
                    if(norm>=Norm){
                        count++;
                        validC[Ci][Cj]++;
                    }
                }
            }
        }
        validRate=(float)count/totalNum;
        // printf("Ti=%d Norm=%f validrate=%f down=%f up=%f",Ti,Norm,validRate,down,up);
        if(abs(validRate-ExpectedRate)<TUNINGERROR) break;
        if(Ti==0){
            Norm = ave/totalNum;
            up=Norm;
            printf("\n");
            continue;
        }
        if(validRate>ExpectedRate){
            // printf(" now>exp ");
            if(up == Norm) up*=2;
            down=Norm;
            Norm = (Norm+up)/2;
            
        }
        else{
            // printf(" now<exp ");
            up=Norm;
            Norm = (Norm+down)/2;
        }
        // printf("==> down=%f up=%f\n",down,up);
    }
    // printf("tuning result: valid mul=%d, simple mul=%d, rate=%f%% aveNorm=%f\n",count,totalNum,(float)count/totalNum*100,ave/totalNum);
    return (float)Norm;

}

//计算有效块乘的个数
void countValid(float* A_normmap,float* B_normmap){
    printf("***counting valid***\n");
    int count=0;
    float norm,ave=0;
    int validC[M/LoNum][N/LoNum];
    
    //直接变成非递归 小块乘
    for(int Ci=0;Ci<M/LoNum;Ci++){
        for(int Cj=0;Cj<N/LoNum;Cj++){
            //计算小块C[Ci,Cj]的结果
            //遍历A,B小块
            validC[Ci][Cj]=0;
            for(int Ki=0;Ki<K/LoNum;Ki++){
                //计算范数乘 小块A[Ci,Ki]和B[Ki,Cj]
                // printf("%f\n",GETELEMENT21(A_normmap,Ci,Ki,K/LoNum));
                norm = GETELEMENT21(A_normmap,Ci,Ki,K/LoNum)*GETELEMENT21(B_normmap,Ki,Cj,N/LoNum);
                ave+=norm;
                // printf("%f\n",norm);
                // printf("%f\n",GETELEMENT21(B_normmap,Ki,Cj,N/LoNum));
                // printf("norm=%f Norm=%f\n",norm,Norm);
                if(norm>=Norm){
                    count++;
                    validC[Ci][Cj]++;
                }
            }
        }
    }

    // MATRIXSHOW21D(A_normmap,M/LoNum,K/LoNum);
    // MATRIXSHOW21D(B_normmap,M/LoNum,K/LoNum);
    // MATRIXSHOW2D(gpu_C,M,N);

    int totalNum=(M/LoNum)*(N/LoNum)*(K/LoNum);
    // CHECKEQ6(C,gpu_C,M,N);
    printf("valid mul=%d, simple mul=%d, rate=%f%% aveNorm=%f\n",count,totalNum,(float)count/totalNum*100,ave/totalNum);

    //把workload写入文件
    // std::ofstream outfile;
    // outfile.open("workload.csv");
    // for(int Ci=0;Ci<M/LoNum;Ci++){
    //     for(int Cj=0;Cj<N/LoNum;Cj++){
    //         outfile<<validC[Ci][Cj];
    //         if(Cj==N/LoNum-1){
    //             outfile<<std::endl;
    //         }
    //         else{
    //             outfile<<",";
    //         }
    //     }
    // }
    // outfile.close();

    //输出X_col的norm map
    std::ofstream outfile;
    outfile.open("X_col.csv");
    for(int i=0;i<K/LoNum;i++){
        for(int j=0;j<N/LoNum;j++){
            outfile<<GETELEMENT21(B_normmap,i,j,N/LoNum);
            if(j==N/LoNum-1){
                outfile<<std::endl;
            }
            else{
                outfile<<",";
            }
        }
    }
    outfile.close();
}

//测试范数,有错不能用
void checkNormMap(float* A,float* A_normmap){
    // printf("***checking norm***\n");
    // int count=0;
    
    // //直接变成非递归 小块乘
    // for(int Ci=0;Ci<M/LoNum;Ci++){
    //     for(int Cj=0;Cj<N/LoNum;Cj++){
    //         //计算小块C[Ci,Cj]的结果
    //         //遍历A小块
    //         for(int Ki=0;Ki<K/LoNum;Ki++){
    //             //计算小块范数A[Ci,Ki]，B[Ki,Cj]
    //             float cpu_norm=0.0f;
    //             //遍历块内元素
    //             for(int i=0;i<LoNum;i++){
    //                 for(int j=0;j<LoNum;j++){
    //                     cpu_norm += A[(Ci*LoNum+i)*N+Ki*LoNum+j] * A[(Ci*LoNum+i)*T+Ki*LoNum+j];
    //                 }
    //             }
    //             cpu_norm=sqrt(cpu_norm);
    //             float gpu_norm = GETELEMENT21(A_normmap,Ci,Ki,K/LoNum);
    //             if(fabs(gpu_norm-cpu_norm)>=E){
    //                 printf("NORM ERROR! (%d,%d) cpu=%f gpu=%f\n",Ci,Cj,cpu_norm,gpu_norm);
    //                 return;
    //             }
    //         }
    //     }
    // }

    // printf("NORM CHECK DONE!\n");

}

//转置B矩阵
mytype* copy_B(mytype* B,int m,int n){
    mytype *b;
    cudaMallocManaged((void **)&b, sizeof(mytype)*m*n);
    // for(int i=0;i<n;i++){
    //     for(int j=0;j<m;j++){
    //         #if USINGHALF
    //         b[i*m+j]=__half2float(B[j*n+i]);
    //         #else
    //         b[i*m+j]=B[j*n+i];
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

//截断
void truncation(float* M, float* ORI,float flag){
    int del=0;
    for(int i=0;i<T;i++){
        for(int j=0;j<T;j++){
            if(fabs(ORI[i*T+j]<flag)){
                M[i*T+j]=0;
                del++;
                // printf("%lf\n",ORI[i*T+j]);
            }
            else{
                M[i*T+j]=ORI[i*T+j];
            }
        }
    }
    printf("spares matrix del=%d valid=%d validNum=%f%%\n",del,T*T-del,(float)(T*T-del)/(T*T)*100);
}

//代数衰减 |a_ij| < c/(|i-j|^v+1); c>0, v>0
void getDecayMatrixAlgDouble(double* A,double c,double v,int m,int n){
    double t;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            t = (double) c / ((double) pow(abs(i-j),v)+1);
            // A[i*n+j] = t;
            A[i*n+j] = t*0.1;
            // if(rand()%2==0) A[i*n+j]*=-1;
        }
    }
}