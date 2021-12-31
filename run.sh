name=cuSpAMM
echo "编译程序"
nvcc -O3 -o main tool.cu $name.cu -w -arch sm_70  -lcublas --std=c++11 -Xcompiler -fopenmp
./main
echo 
echo "------------------------"

# name=cusparse
# nvcc -O3 -o cusparse $name.cu -w -arch sm_70  -lcublas -lcusparse --std=c++11 -Xcompiler -fopenmp
# ./cusparse
