#define USINGHALF 0
const double NormINIT=1.48
;
#define TESTTIME 20
#define WARMUP 3
const int inM=1024;
const int inK=1024;
const int inN=1024;
#define DEVICEDIM 2
#define CNN 0
#define DECAY 0
#define MATRIXNOR 0
#define MATRIXEXP 0
#define MATRIXALG 1
#if DECAY
const std::string FILENAMEA="a";
const std::string FILENAMEB="b";
#endif
#if CNN
const std::string FILENAMEA="data_cnn/conv_w_col.csv(64, 576).csv";
const std::string FILENAMEB="data_cnn/conv_X_col.csv(576, 102400).csv";
#endif
