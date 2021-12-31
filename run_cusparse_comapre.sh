
name=cusparse_compare
nvcc -O3 -o cusparse $name.cu tool.cu -w -arch sm_70  -lcublas -lcusparse --std=c++11 -Xcompiler -fopenmp
./cusparse
