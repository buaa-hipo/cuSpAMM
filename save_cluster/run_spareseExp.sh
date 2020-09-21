nvcc sparseMatrixExp.cu tool.cu -lcusparse -lcublas -o main
CUDA_VISIBLE_DEVICES=1 ./main
echo
echo