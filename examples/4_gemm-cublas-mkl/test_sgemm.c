#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include <cublas.h>
#include <mkl.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// basic function to compare two arrays
float get_max_error(float* ref, float *res, int n, int inc)
{
	int i;
	float max_err = -1.0;
	float err = -1.0;
	inc = abs(inc);
	for(i = 0; i < n; i++)
	{
		err = fabs(res[i * inc] - ref[i * inc]);
		if(ref[i * inc] != 0.0)err /= fabs(ref[i * inc]);
		if(err > max_err)max_err = err;
	}
	return max_err;
}


// CPU code
int main(int argc, char* argv[])
{
	int dim = 1024 + 10 * 1024;
	
	int matrix_size = dim * dim;
	
	// cpu pointers
	float *ha, *hb, *hc_cpu, *hc_gpu;
	
	// gpu pointers
	float *da, *db, *dc;
	
	// cuda events for timing
	cudaEvent_t start, stop;
	
	//elapsed time
	float gpu_time = 0.0;
	double cpu_time = 0.0;
	
	// create cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("Allocating matrices on CPU ... \n");
	// allocate and initialize cpu memory
	ha = (float*)malloc(matrix_size * sizeof(float));
	hb = (float*)malloc(matrix_size * sizeof(float));
	hc_cpu = (float*)malloc(matrix_size * sizeof(float));
	hc_gpu = (float*)malloc(matrix_size * sizeof(float));
	
	printf("Initializing matrices ... \n");
	// init matrices randomly
	for(int i = 0; i < matrix_size; i++)
	{
		ha[i] = (float)rand() / (float)RAND_MAX;
		hb[i] = (float)rand() / (float)RAND_MAX;
	}
	memset(hc_cpu, 0, matrix_size * sizeof(float));			// init hc with zeros
	memset(hc_gpu, 0, matrix_size * sizeof(float));			// init hc with zeros
	
	
	printf("Allocating matrices on GPU ... \n");
	// allocate gpu memory
	cudaMalloc((void**)&da, matrix_size * sizeof(float));
	cudaMalloc((void**)&db, matrix_size * sizeof(float));
	cudaMalloc((void**)&dc, matrix_size * sizeof(float));
	
	printf("Offloading input to GPU ... \n");
	// memory copy host to device
	cudaMemcpy(da, ha, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dc, hc_gpu, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
	
	// printing the result header lines
	{
		printf("-------------------------- TESTING SGEMM -------------------------\n");
		printf("==================================================================\n\n");
		printf(" Matrix Dim      CUBLAS (Gflop/s)     MKL (Gflop/s)     Max. Error\n");
		printf("------------     ----------------     -------------     ----------\n");
	}
	
	// beginning test
	{
		int start_dim = 1024; int end_dim = dim, increment = 1024;
		
		
		for(int i = start_dim; i <= end_dim ; i+= increment)
		{
			printf("%-12d     ", i);
			int m = i;
			
			float dimf = (float)m;
			float million_flops = 2 * 1e-6 * dimf * dimf * dimf;	// 2 * n ^ 3
	
			// gpu sgemm (cublas)
			cudaMemset(dc, 0, matrix_size * sizeof(float));
			cudaEventRecord(start, 0);
			cublasSgemm('N', 'N', m, m, m, 1.0, da, m, db, m, 0.0,  dc, m);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&gpu_time, start, stop);
			float gpu_perf = million_flops / gpu_time; 
			printf("%-16.2f     ", gpu_perf);
			
			// cpu sgemm (mkl)
			memset(hc_cpu, 0, matrix_size * sizeof(float));
			struct timeval start, end;
			gettimeofday(&start, NULL);
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, ha, m, hb, m, 0.0, hc_cpu, m);
  			gettimeofday(&end, NULL);
  			cpu_time = (end.tv_sec - start.tv_sec) * 1000;
  			cpu_time += (end.tv_usec - start.tv_usec) * 1e-3;
  			float cpu_perf = million_flops / cpu_time; 
			printf("%-13.2f     ", cpu_perf);
			
			// compute error
			int size_ = m * m;
			cudaMemcpy(hc_gpu, dc, size_ * sizeof(float), cudaMemcpyDeviceToHost);
			
			float err = get_max_error(hc_cpu, hc_gpu, size_, 1);
			printf("%-10.2e     ", err);
			
			printf("\n");
		}
	}
	
	// destroy cuda events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// free cpu memory
	if(ha)free(ha);
	if(hb)free(hb);
	if(hc_cpu)free(hc_cpu);
	if(hc_gpu)free(hc_gpu);
	
	// free gpu memory
	if(da)cudaFree(da);
	if(db)cudaFree(db);
	if(dc)cudaFree(dc);
	
	return 0;	
}