#define USINGHALF 1
const double NormINIT=1.548969
;
#define TESTTIME 10
#define WARMUP 5
const int inM=2048;
const int inK=2048;
const int inN=2048;
#define DEVICEDIM 1
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
