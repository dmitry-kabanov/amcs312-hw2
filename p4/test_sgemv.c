#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "kblas.h"


#define FMULS_GEMV(n) ((n) * (n) + 2. * (n))
#define FADDS_GEMV(n) ((n) * (n)           )

#define PRECISION_s

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_GEMV(n) + 2. * FADDS_GEMV(n))
#else
#define FLOPS(n) (      FMULS_GEMV(n) +      FADDS_GEMV(n))
#endif

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

int main(int argc, char** argv)
{
	if(argc < 3)
	{printf("USAGE: %s <device-id> <no-trnas'n'-or-trans't'>\n", argv[0]); exit(-1);}
	
	int dev = atoi(argv[1]);
	char trans = *argv[2];
	cudaSetDevice(dev);
	
	cublasHandle_t cublas_handle;
	cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
	cublasCreate(&cublas_handle);
	cublasSetAtomicsMode(cublas_handle, mode);
	
	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	
    int M = 241 * 64;
    int N = M;
    int LDA = ((M+31)/32)*32;

	int incx = 1;
	int incy = 1;
	int vecsize_x = N * abs(incx);
	int vecsize_y = M * abs(incy);
	
	cublasOperation_t trans_;
	if(trans == 'N' || trans == 'n')
		trans_ = CUBLAS_OP_N;
	else if (trans == 'T' || trans == 't')
		trans_ = CUBLAS_OP_T;
		
	float alpha = 2.3, beta = -0.6;
	
	cudaError_t err;
	cudaEvent_t start, stop; 
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    // point to host memory
    float* A = NULL;
    float* x = NULL;
    float* ycuda = NULL;
    float* ykblas = NULL;
	
    // point to device memory
    float* dA = NULL;
    float* dx = NULL;
    float* dy = NULL;

    if(trans == 'N' || trans == 'n')printf("non-transposed test .. \n");
	else if (trans == 'T' || trans == 't') printf("transposed test .. \n");
	else { printf("transpose modes is not properly specified\n"); exit(-1);}
	printf("Allocating Matrices\n");
    A = (float*)malloc(M*N*sizeof(float));
    x = (float*)malloc(vecsize_x*sizeof(float));
    ycuda = (float*)malloc(vecsize_y*sizeof(float));
    ykblas = (float*)malloc(vecsize_y*sizeof(float));
    
    cudaMalloc((void**)&dA, M*N*sizeof(float));
    cudaMalloc((void**)&dx, vecsize_x*sizeof(float));
    cudaMalloc((void**)&dy, vecsize_y*sizeof(float));

    // Initialize matrix and vector
    int i, j, m;
    for(i = 0; i < M * N; i++)
      A[i] = ( (float)rand() ) / (float)RAND_MAX;
      
    for(i = 0; i < vecsize_x; i++)
      x[i] = ( (float)rand() ) / (float)RAND_MAX;
    
    cudaMemcpy(dA, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, vecsize_x*sizeof(float), cudaMemcpyHostToDevice);

	int write_to_file = 0;
	FILE* fp;
	if(write_to_file)
	{
		// prepare a matlab file to produce the output
		char file_name[100] = "ssymv_perf_";
		if(deviceProp.major > 2)strcat(file_name, "kepler.m");
		else  strcat(file_name, "fermi.m");
		fp = fopen (file_name,"w");
	}
	
    printf("------------------- Testing SSYMV ----------------\n");
    printf("  Matrix         CUDA        KBLAS          Max.  \n");
    printf(" Dimension     (Gflop/s)   (Gflop/s)       Error  \n");
    printf("-----------   ----------   ----------   ----------\n");
    
    if(write_to_file)fprintf(fp, "result = [\n");
    
    for(m = 64; m < M; m *= 2)
    {
    	float elapsedTime; 
    	
      	int lda = ((m+31)/32)*32;
      	float flops = FLOPS( (float)m ) / 1e6;
		
		cublasSetMatrix(m, m, sizeof(float), A, LDA, dA, lda);
		
		for(i = 0; i < vecsize_y; i++)
    	{
      		ycuda[i] = ( (float)rand() ) / (float)RAND_MAX;
      		ykblas[i] = ycuda[i];
    	}
      
      	// --- cuda test
      	cudaMemcpy(dy, ycuda, vecsize_y * sizeof(float), cudaMemcpyHostToDevice);
      
      	cudaEventRecord(start, 0);
      	cublasSgemv(cublas_handle, trans_, m, m, &alpha, dA, lda, dx, incx, &beta, dy, incy);
      	cudaEventRecord(stop, 0);
      	cudaEventSynchronize(stop);
      
      	elapsedTime = 0.0;
      	cudaEventElapsedTime(&elapsedTime, start, stop);
      	float cuda_perf = flops / elapsedTime;
      
      	cudaMemcpy(ycuda, dy, vecsize_y * sizeof(float), cudaMemcpyDeviceToHost);
      	// end of cuda test
      	  	
      	// ---- kblas
      	cudaMemcpy(dy, ykblas, vecsize_y * sizeof(float), cudaMemcpyHostToDevice);
      	
      	cudaEventRecord(start, 0);
      	
      	if(deviceProp.major > 2)
      			kblas_sgemv_kepler( trans, m, m, alpha, dA, lda, dx, incx, beta, dy, incy);
      	else 	
      			kblas_sgemv_fermi( trans, m, m, alpha, dA, lda, dx, incx, beta, dy, incy);
      	
      	cudaEventRecord(stop, 0);
      	cudaEventSynchronize(stop);
      
      	cudaMemcpy(ykblas, dy, vecsize_y * sizeof(float), cudaMemcpyDeviceToHost);
      
      	elapsedTime = 0.0;
      	cudaEventElapsedTime(&elapsedTime, start, stop);
      	float kblas_perf = flops / elapsedTime;

      	
      	// testing error -- specify ref. vector and result vector
      	float* yref = ycuda; 
      	float* yres = ykblas;
      	
      	float error = get_max_error(yref, yres, m, incy);
      
      	//printf("-----------   ----------   ----------   ----------   ----------   ----------\n");
    	printf("%-11d   %-10.2f   %-10.2f   %-10e;\n", m, cuda_perf, kblas_perf, error);
    	if(write_to_file)fprintf(fp, "%-11d   %-10.2f   %-10.2f   %-10e;\n", m, cuda_perf, kblas_perf, error);
    }
	if(write_to_file)fprintf(fp, "];\n");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
    if(dA)cudaFree(dA);
    if(dx)cudaFree(dx);
    if(dy)cudaFree(dy);
    
    if(A)free(A);
    if(x)free(x);
    if(ycuda)free(ycuda);
	if(ykblas)free(ykblas);

	if(write_to_file)
	{
		// write some matlab code to the file
		fprintf(fp, "dim = result(:, 1);\n");
		fprintf(fp, "cublas = result(:, 2);\n");
		fprintf(fp, "kblas = result(:, 3);\n");
		fprintf(fp, "error = result(:, 4);\n");
		
		fprintf(fp, "h=figure; hold on; grid on;\n");
		fprintf(fp,"plot(dim, cublas, 'b', 'LineWidth',2);\n");
		fprintf(fp,"plot(dim, kblas, 'r', 'LineWidth',2);\n");
		fprintf(fp,"legend('cublas-5.0', 'KBLAS');\n");
		fprintf(fp,"set(h, 'Position', [1050 554 750 0.75*750]);\n");
		
		fprintf(fp,"xlabel('Matrix dimension');\n");
		fprintf(fp,"ylabel('Gflop/s');\n");
		if(deviceProp.major > 2)
		{
			if(trans == 'N' || trans == 'n')fprintf(fp,"title('SGEMV-N Performance on Kepler');\n");
			else fprintf(fp,"title('SGEMV-N Performance on Kepler');\n");
		}
		else
		{
			if(trans == 'T' || trans == 't')fprintf(fp,"title('SGEMV-T Performance on Fermi');\n");
			else fprintf(fp,"title('SGEMV-T Performance on Fermi');\n");
		}
		// end
		// end of matlab code
		if(fp)fclose(fp);
	}
	cublasDestroy(cublas_handle);
    return EXIT_SUCCESS;
}

