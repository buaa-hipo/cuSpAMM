name=getAveNorm
nvcc -O3 -o main tool.cu $name.cu -w -arch sm_70  -lcublas
# cuda-memcheck main
# nvprof ./main
#./main > out
./main 
# sudo nvprof --metrics achieved_occupancy  ./main
echo $name
echo 
echo

