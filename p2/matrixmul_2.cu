#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

template<int block_size>
__global__
void matrixmul_2(float* A, float* B, float* C, int dim)
{
	// constant block/thread indices
	const int bx = blockIdx.x;
	const int by = blockIdx.y;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	__shared__ float as[block_size][block_size];
	__shared__ float bs[block_size][block_size];
	
	// each thread is responsible for one element in the output matrix
	// (row, col) indices of the output element 
	const int row = by * block_size + ty;
	const int col = bx * block_size + tx;
	
	volatile int tile = block_size;		// being volatile prevents unrolling the inner-most loop
	
	float sum = 0.0;
	for(int m = 0; m < dim/block_size; m++)
	{
		as[ty][tx] = A[row * dim + (m * block_size + tx)];
		bs[ty][tx] = B[(m * block_size + ty) * dim + col];
		
		__syncthreads();
		
		for(int k = 0; k < tile; k++)
			sum += as[ty][k] * bs[k][tx]; 
		
		__syncthreads();
	}
	C[row * dim + col] = sum;
}

extern "C"
void matrixmul_2_driver(float* dA, float* dB, float* dC, int dim, int block_size)
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
	
	//printf("block(%d, %d)\n", dimGrid.x, dimGrid.y);
	// launch the kernel .. the template parameter must be known at compile time
	switch(block_size)
	{
		case  1: matrixmul_2< 1><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case  2: matrixmul_2< 2><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case  4: matrixmul_2< 4><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case  8: matrixmul_2< 8><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case 16: matrixmul_2<16><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case 32: matrixmul_2<32><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		default:{printf("ERROR: block_size is not supported\n"); return;}
	}
	
}