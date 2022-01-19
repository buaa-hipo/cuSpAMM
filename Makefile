NVCC=nvcc
FLAGS= --std=c++11

cublas:
	${NVCC} ${FLAGS} -lcublas test_cublas.cu -o test_cublas

cublasxt:
	${NVCC} ${FLAGS} -lcublas test_cublasxt.cu -o test_cublasxt
	
cusparse:
	${NVCC} ${FLAGS} -lcusparse test_cusparse.cu -o test_cusparse

cusparse_cublas:
	${NVCC} ${FLAGS} -lcublas test_cusparse_cublas.cu -o test_cusparse_cublas

cusparse_real:
	${NVCC} ${FLAGS} -lcublas -lcusparse test_cusparse_real.cu -o test_cusparse_real


cu10_cusparse:
	${NVCC} ${FLAGS} -lcublas -lcusparse test_cu10_cusparse.cu -o test_cu10_cusparse