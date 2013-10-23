#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include <cublas.h>
//#include <mkl.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "matrixmul.h"

// CPU code
int main(int argc, char* argv[])
{
	if(argc < 6)
	{
		printf("\nUSAGE: %s <matrix-dim> <run-kernel-1> <run-kernel-2> <run-kernel-3> <run-kernel-4>\n", argv[0]);
		exit(-1);
	}
	int dim = atoi( argv[1] );	// matrix dimension
	if(dim < 128)
	{
		printf("\nmatrix dimension should be 64 at minimum ..\ncontinuing with dimension set to 128 .. \n");
		dim = 128;
	}
		
	// for simplicity dim has to be divisible by 32
	{
		int dim_ = ( (dim + 31) / 32 ) * 32;
		if(dim != dim_)
		{
			printf("\nMatrix dim %d has to be multiple of 32 .. wrapping up to %d\n", dim, dim_);
			dim  = dim_;
		}
	}
	int run_kernel[4];
	run_kernel[0] = abs( atoi( argv[2] ) );
	run_kernel[1] = abs( atoi( argv[3] ) );
	run_kernel[2] = abs( atoi( argv[4] ) );
	run_kernel[3] = abs( atoi( argv[5] ) );
	
	// fast return check
	{
		int sum = 0;
		for(int ii = 0; ii < 4; ii++)sum += run_kernel[ii];
		if(sum == 0) 
		{
			printf("\n No kernel is selected .. exiting!\n");
			exit(-1);
		}
	}
	
	printf("\n");
	
	// block sizes for testing
	// Note that for Fermi gpus, the 32 block size requires 1024 threads per thread-block
	// which is not supported on Femri
	int block_sizes[6] = {1, 2, 4, 8, 16, 32};
	
	int matrix_size = dim * dim;
	
	printf("Running kernel(s) # ");
	int tmp = 1;
	for(int i = 0; i < 4; i++)
	{
		if(run_kernel[i] == 0)continue;
		if(tmp != 1)printf(" ,"); else tmp = 0;
		printf("%d", i+1);
	}
	printf("\n\n");
	
	// cpu pointers
	float *ha, *hx, *hy;
	// gpu pointers
	float *da, *dx, *dy;
	
	// cuda events for timing
	cudaEvent_t start, stop;
	
	//elapsed time
	float gpu_time = 0.0;
	double cpu_time = 0.0;
	
	// computing flops for measuring performance
	float dimf = (float)dim;
	float million_flops = 2 * 1e-6 * dimf * dimf * dimf;	// 2 * n ^ 3
	
	// create cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("Allocating matrices on CPU ... \n");
	// allocate and initialize cpu memory
	ha = (float*)malloc(matrix_size * sizeof(float));
	hx = (float*)malloc(dim * sizeof(float));
	hy = (float*)malloc(dim * sizeof(float));
	
	printf("Initializing matrix ... \n");
	// init matrices randomly
	for(int i = 0; i < matrix_size; i++)
	{
		ha[i] = (float)rand() / (float)RAND_MAX;
	}
	// init vector x randomly
	for(int i = 0; i < dim; i++)
	{
		hx[i] = (float)rand() / (float)RAND_MAX;
	}
	memset(hy, 0, dim * sizeof(float));			// init hy with zeros
	
	
	printf("Allocating memory on GPU ... \n");
	// allocate gpu memory
	cudaMalloc((void**)&da, matrix_size * sizeof(float));
	cudaMalloc((void**)&dx, dim * sizeof(float));
	cudaMalloc((void**)&dy, dim * sizeof(float));
	
	printf("Offloading input to GPU ... \n");
	// memory copy host to device
	cudaMemcpy(da, ha, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dx, hx, dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dy, hy, dim * sizeof(float), cudaMemcpyHostToDevice);
	
	// printing the result header lines
	{
		printf("Matrix (vector) dimension = %d\n", dim);
		printf("---------------------\n");
		printf(" Block     ");
		for(int t = 0; t < 4; t++)
		{ 
			if(run_kernel[t] != 0) 
				printf("kernel %d      ", t+1);
		}
		printf("\n");
		printf(" Size      ");
		for(int t = 0; t < 4; t++) if(run_kernel[t] != 0) printf("(Gflop/s)     ");
		printf("\n");
		printf("------     ");
		for(int t = 0; t < 4; t++) if(run_kernel[t] != 0) printf("---------     ");
		printf("\n");
	} // finish printing the header lines
	
	// there is an issue with cublasSgemm .. scores bad performance in the first call
	if(1) cublasSgemm('N', 'N', 1, 1, 1, 1.0, da, 1, dx, 1, 0.0,  dy, 1);
	
	// loop over block sizes
	for(int i = 0; i < 6; i++)
	{
		int block_size = block_sizes[i]; 
		printf("%-6d     ", block_size);
		
		// loop over kernels
		for(int j = 0; j < 4; j++)
		{
			if(run_kernel[j] == 0) continue;
			
			cudaMemset(dy, 0, matrix_size * sizeof(float));
			cudaEventRecord(start, 0);
			switch(j)
			{
				case 0:matrixmul_1_driver(da, dx, dy, dim, block_size); break;
				case 1:matrixmul_2_driver(da, dx, dy, dim, block_size); break;
				case 2:matrixmul_3_driver(da, dx, dy, dim, block_size); break;
				case 3:matrixmul_4_driver(da, dx, dy, dim, block_size); break;
				case 4:cublasSgemm('N', 'N', dim, dim, dim, 1.0, da, dim, dx, dim, 0.0,  dy, dim); break;
				default:;
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&gpu_time, start, stop);
			float gpu_perf = million_flops / gpu_time; 
			printf("%-9.2f     ", gpu_perf);
			
			// should read result back
			//}
			//else if(j == 5) 	// mkl gemm
			//{
			//	struct timeval start, end;
			//	gettimeofday(&start, NULL);
			//	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, ha, dim, hx, dim, 0.0, hy, dim);
  			//	gettimeofday(&end, NULL);
  			//	cpu_time = (end.tv_sec - start.tv_sec) * 1000;
  			//	cpu_time += (end.tv_usec - start.tv_usec) * 1e-3;
  			//	float cpu_perf = million_flops / cpu_time; 
			//	printf("%-9.2f     ", cpu_perf);
		}
		printf("\n");
	}
	
	// destroy cuda events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// free cpu memory
	if(ha)free(ha);
	if(hx)free(hx);
	if(hy)free(hy);
	
	// free gpu memory
	if(da)cudaFree(da);
	if(dx)cudaFree(dx);
	if(dy)cudaFree(dy);
	
	return 0;	
}
