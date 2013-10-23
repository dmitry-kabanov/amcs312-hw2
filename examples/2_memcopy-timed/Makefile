# cuda root
_CUDA_ROOT_ = /usr/local/cuda

NVCC = nvcc
# include and lib paths
INCLUDES=-I${_CUDA_ROOT_}/include
LIB_PATH=-L${_CUDA_ROOT_}/lib64

# libraries to link against
LIB= -lcudart -lcublas

CU_SRC=	memcopy-timed.cu
		
EXE=$(CU_SRC:.cu=)
#------------------------------
# Choose your gpu arch

#kepler GK 110
SM = sm_35

#kepler GK104
#SM = sm_30

#fermi
#SM = sm_20
#------------------------------
all: $(EXE)

$(EXE): $(CU_SRC)
	$(NVCC) -arch $(SM) $(CU_SRC) -o $(EXE) $(LIB_PATH) $(LIB)

clean:
	rm -f *.o *.cu_o $(EXE)

