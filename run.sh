#name=baseline
name=single
#nvcc -O3 -o main tool.cu $name.cu -w -arch sm_70 
nvcc -O3 -o main tool.cu $name.cu -w -arch sm_70  -lcublas --std=c++11 
# cuda-memcheck main
# nvprof ./main
#./main > out
./main 
# sudo nvprof --metrics achieved_occupancy  ./main
echo $name
echo 
echo
