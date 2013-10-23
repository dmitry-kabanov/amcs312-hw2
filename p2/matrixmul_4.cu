#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

template<int block_size>
__global__
void matrixmu_4(float* A, float* B, float* C, int dim)
{
	// constant block/thread indices
	const int bx = blockIdx.x;
	const int by = blockIdx.y;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	
	float reg_a = 0.0, reg_b = 0.0;
	
	__shared__ float as[block_size][block_size];
	__shared__ float bs[block_size][block_size];
		
	// each thread is responsible for one element in the output matrix
	// (row, col) indices of the output element 
	const int row = by * block_size + ty;
	const int col = bx * block_size + tx;
	
	float sum = 0.0;
	// load a tile into registers
	reg_a = A[row * dim + (0 * block_size + tx)];
	reg_b = B[(0 * block_size + ty) * dim + col];
	
	for(int m = 1; m < dim/block_size; m++)
	{
		// offload tile from registers to shmem
		as[ty][tx] = reg_a;
		bs[ty][tx] = reg_b;
		
		__syncthreads();
		
		// load next tile
		reg_a = A[row * dim + (m * block_size + tx)];
		reg_b = B[(m * block_size + ty) * dim + col];
		
		// compute current tile
		#pragma unroll
		for(int k = 0; k < block_size; k++)
			sum += as[ty][k] * bs[k][tx]; 
		__syncthreads();
	}
	
	// offload tile from registers to shmem
	as[ty][tx] = reg_a;
	bs[ty][tx] = reg_b;
		
	__syncthreads();
		
	// compute current tile
	#pragma unroll
	for(int k = 0; k < block_size; k++)
		sum += as[ty][k] * bs[k][tx]; 
		
	C[row * dim + col] = sum;
}

extern "C"
void matrixmul_4_driver(float* dA, float* dB, float* dC, int dim, int block_size)
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
		case  1: matrixmu_4< 1><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case  2: matrixmu_4< 2><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case  4: matrixmu_4< 4><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case  8: matrixmu_4< 8><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case 16: matrixmu_4<16><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		case 32: matrixmu_4<32><<<dimGrid, dimBlock>>>(dA, dB, dC, dim); break;
		default:{printf("ERROR: block_size is not supported\n"); return;}
	}
	
}