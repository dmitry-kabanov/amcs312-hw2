# cuda root .. change as needed
_CUDA_ROOT_ = /usr/local/cuda

NVCC=nvcc
CC=gcc

# include and lib paths
INCLUDES=-I${_CUDA_ROOT_}/include
LIB_PATH=-L${_CUDA_ROOT_}/lib64

# libraries to link against
LIB= -lcudart -lcublas

CU_SRC=matrixvecmul_1.cu matrixmul_2.cu matrixvecmul_3.cu matrixmul_4.cu
C_SRC=test_matrixvecmul.c

CU_OBJ=$(CU_SRC:.cu=.cu_o)
C_OBJ=$(C_SRC:.c=.o)
ALL_OBJ=$(CU_OBJ) $(C_OBJ)

EXE=$(C_SRC:.c=)

#------------------------------
# Choose your gpu arch
#kepler GK 110
SM = sm_35
#kepler GK104
#SM = sm_30
#fermi
#SM = sm_20
#------------------------------

%.cu_o: %.cu
	$(NVCC) -arch $(SM) $(NVOPTS) -c $< -o $@

%.o: %.c
	$(CC) -std=c99 -c $(INCLUDES) $< -o $@ 

$(EXE): $(CU_OBJ) $(C_OBJ)
	$(CC) $(CU_OBJ) $(C_OBJ) -o $(EXE) $(LIB_PATH) $(LIB)

all: $(EXE)

clean:
	rm -f *.o *.cu_o $(EXE)

