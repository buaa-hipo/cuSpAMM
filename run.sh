# name=single
name=single
#nvcc -O3 -o main tool.cu $name.cu -w -arch sm_70 
nvcc -O3 -o main tool.cu $name.cu -w -arch sm_70  -lcublas --std=c++11 -Xcompiler -fopenmp
# nvcc -O3 -o main $name.cu -w -arch sm_70  -lcublas --std=c++11 -Xcompiler -fopenmp

# cuda-memcheck main
# nvprof ./main
#./main > out
CUDA_VISIBLE_DEVICES=1 ./main
# ./main 
# sudo nvprof --metrics achieved_occupancy  ./main
echo $name
echo 
echo

