#define USINGHALF 0
<<<<<<< HEAD
const double NormINIT=0.01;
#define TESTTIME 20
#define WARMUP 3
const int inM=13656;
const int inK=13656;
const int inN=13656;
#define DEVICEDIM 2
=======
const double NormINIT=0.92 
;
#define TESTTIME 20
#define WARMUP 3
const int inM=32768;
const int inK=32768;
const int inN=32768;
#define DEVICEDIM 1
>>>>>>> d60a1185cffc63aea87fdd2c9f5cd4fbfe107d83
#define CNN 0
#define DECAY 1
#define MATRIXNOR 0
#define MATRIXEXP 0
#define MATRIXALG 0
#if DECAY
const std::string FILENAMEA="data_decay/S_matrix_13656.mtx";
const std::string FILENAMEB="data_decay/S_matrix_13656.mtx";
#endif
#if CNN
const std::string FILENAMEA="data_cnn/conv_w_col.csv(64, 576).csv";
const std::string FILENAMEB="data_cnn/conv_X_col.csv(576, 102400).csv";
#endif
