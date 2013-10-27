# ROOTS
_CUDA_ROOT_ = /usr/local/cuda
_KBLAS_ROOT_ = /usr/local/kblas

# include and lib paths
INCLUDES=-I${_CUDA_ROOT_}/include -I${_KBLAS_ROOT_}/include
LIB_PATH=-L${_CUDA_ROOT_}/lib64 -L${_KBLAS_ROOT_}/lib

# libraries to link against
LIB = -lcublas -lcudart -lkblas_fermi -lkblas_kepler

C_SRC = test_sgemv.c
		
C_OBJ=$(C_SRC:.c=.o)
EXE=$(C_SRC:.c=)

%.o: %.c
	gcc -std=c99 $(CFLAGS) $(INCLUDES) -c $< -o $@ 

test_%: test_%.o
	gcc $< -o $@ $(LIB_PATH) $(LIB)

all: $(EXE)

$(EXE): $(C_OBJ)

clean:
	rm -f *.o *.m $(EXE)

