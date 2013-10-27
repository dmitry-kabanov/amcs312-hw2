#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__
void matrixmul_1(float* A, float* x, float* y, int dim, int block_size)
{
	const int row = blockIdx.y * block_size + threadIdx.y;
	
	float sum = 0.0;
	for(int k = 0; k < dim; k++)
		sum += A[row * dim + k] * x[k];
	
	y[row] = sum;
}

extern "C"
void matrixmul_1_driver(float* dA, float* dx, float* dy, int dim, int block_size)
{
	if(dim % block_size != 0){printf("ERROR: Block size does not fully divide matrix dimension .. exiting\n"); return;}
	
	// thread block configuration
	int block_dim_x = block_size;
	int block_dim_y = block_size;
	
	// kernel grid configuration 
	int grid_dim_x = dim / block_size;
	int grid_dim_y = dim / block_size;

	dim3 dimBlock(block_dim_x, block_dim_y);
	dim3 dimGrid(grid_dim_x, grid_dim_y);
	
	matrixmul_1<<<dimGrid, dimBlock>>>(dA, dx, dy, dim, block_size);
}
