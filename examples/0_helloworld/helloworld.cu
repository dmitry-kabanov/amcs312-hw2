#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>

// GPU code
__global__ 
void helloworld_1()
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;
	
	printf("Hello from thread (%d, %d, %d) in block (%d, %d, %d)\n", tx, ty, tz, bx, by, bz);
}

__global__ 
void helloworld_2()
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int tid = ty * blockDim.x + tx;
	printf("Hello from thread (%d, %d) => %d \n", tx, ty, tid);
}

// CPU code
int main(int argc, char* argv[])
{
	// read input parameters
	if(argc < 4)
	{
		printf("USAGE: %s <block-dim-x> <block-dim-y> <block-dim-z>\n", argv[0]);
		exit(-1);
	} 
	
	int block_dim_x = atoi( argv[1] );
	int block_dim_y = atoi( argv[2] );
	int block_dim_z = atoi( argv[3] );
	
	dim3 block_1(block_dim_x, block_dim_y, block_dim_z);
	
	dim3 block_2(block_dim_x, block_dim_y);
	
	// in both kernels, the grid has one thread block only
	dim3 grid(1); 
	
	printf("Starting kernel execution .. \n");
	printf("-----------------------------\n");
	
	// no input data required .. just run the kernels
	printf("kernel 1:\n==========\n");
	helloworld_1<<<grid, block_1>>>();
	
	// wait for kernel to finish
	cudaDeviceSynchronize();
	printf("\n");
	
	printf("kernel 2:\n==========\n");
	helloworld_2<<<grid, block_2>>>();
	
	// wait for kernel to finish
	cudaDeviceSynchronize();
	
	return 0;
}