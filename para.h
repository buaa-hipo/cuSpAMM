#define USINGHALF 0
const double NormINIT=0;
#define TESTTIME 10
#define WARMUP 3
const int inM=4096;
const int inK=4096;
const int inN=4096;
#define DEVICEDIM 1
#define CNN 0
#define DECAY 0
#define MATRIXNOR 0
#define MATRIXEXP 0
#define MATRIXALG 1
#if DECAY
#endif
#if CNN
const std::string FILENAMEA="data_cnn/conv_w_col.csv(64, 576).csv";
const std::string FILENAMEB="data_cnn/conv_X_col.csv(576, 102400).csv";
#endif
