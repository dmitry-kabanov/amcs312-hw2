# cuda root .. change as needed
_CUDA_ROOT_ = /usr/local/cuda

# mkl root .. change as needed
_MKL_ROOT_ = /opt/intel/composer_xe_2013.1.117/mkl
_ICC_ROOT_ = /opt/intel/composer_xe_2013.1.117/compiler

NVCC=nvcc
CC=gcc
CCOPTS= -DMKL_ILP64 -m64

# include and lib paths
INCLUDES=-I${_CUDA_ROOT_}/include -I$(_MKL_ROOT_)/include
LIB_PATH=-L${_CUDA_ROOT_}/lib64 -L$(_MKL_ROOT_)/lib/intel64 -L$(_ICC_ROOT_)/lib/intel64

# libraries to link against
LIB= -lcudart -lcublas

# mkl multi threaded
LIB+=-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm

# mkl sequential
#LIB+=-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm

C_SRC=test_sgemm.c

C_OBJ=$(C_SRC:.c=.o)
ALL_OBJ=$(C_OBJ)

EXE=$(C_SRC:.c=)

%.o: %.c
	$(CC) -std=c99 -c $(CCOPTS) $(INCLUDES) $< -o $@ 

$(EXE): $(C_OBJ)
	$(CC) $(C_OBJ) -o $(EXE) $(LIB_PATH) $(LIB)

all: $(EXE)

clean:
	rm -f *.o $(EXE)

