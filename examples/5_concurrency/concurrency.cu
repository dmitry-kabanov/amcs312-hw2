#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define block_size		(512 * 512)
#define num_blocks		(4)
#define thread_x		(512)
#define num_iterations	(block_size / thread_x)



__global__ 
void process_1(float* in, float threshold, float* out)
{
	const int tx = threadIdx.x; 
	
	#pragma unroll
	for(int i = 0; i < block_size/thread_x; i++)
	{
		if(in[tx] > threshold)
			out[tx] = threshold;
		else
			out[tx] = in[tx];
		in  += thread_x;
		out += thread_x;
	}

}

__global__
void process_2(float* in1, float* in2, float* out)
{
	const int tx = threadIdx.x;
	
	#pragma unroll
	for(int i = 0; i < block_size/thread_x; i++)
	{
		out[tx] = (in1[tx] + in2[tx]) * (in1[thread_x-1 - tx] - in2[thread_x-1 - tx]);
		in1 += thread_x;
		in2 += thread_x;
		out += thread_x;
	}
}

__global__ 
void postprocess(float* out1, float* out2, float* final_out)
{
	const int tx = threadIdx.x;
	#pragma unroll
	for(int i = 0; i < block_size/thread_x; i++)
	{
		final_out[tx] = out1[tx] * out2[tx];
		out1 += thread_x;
		out2 += thread_x;
		final_out += thread_x;
	} 
}

void compute_error(float* ref, float* res, int length, float* max_error, int* index)
{
	float max_err; 
	max_err = fabs(ref[0] - res[0]);
	if(ref[0] != 0)max_err /= fabs(ref[0]);
	int ind = 0;

	for(int i = 1; i < length; i++)
	{
		float err = fabs(ref[i] - res[i]);
		if(ref[i] != 0)err /= fabs(ref[i]);

		if(err > max_err)
		{
			max_err = err;
			ind = i;
		}
	}
	*max_error = max_err;
	*index = ind;
}

void compute_cpu(float* in1_h, float* in2_h, float threshold, float* out1_h, float* out2_h, float* out_h)
{
	float *in1 = in1_h, *in2 = in2_h, *out1 = out1_h, *out2 = out2_h, *out = out_h;
	for(int b = 0; b < num_blocks; b++)
		for(int i = 0; i < num_iterations; i++)
		{
			for(int k = 0; k < thread_x; k++)
			{
				out1[k] = (in1[k] > threshold) ? threshold : in1[k];
				out2[k] = (in1[k] + in2[k]) * (in1[thread_x-1-k] - in2[thread_x-1-k]);
				out[k] = out1[k] * out2[k];
			}
			in1 += thread_x;
			in2 += thread_x;
			out1 += thread_x;
			out2 += thread_x;
			out += thread_x;
		}	
}
	
void compute_gpu_1(float* in1_d, float* in2_d, float threshold, float* out1_d, float* out2_d, float* out_d, float* out_h, float* elapsed_time)
{
	cudaEvent_t start, stop; 
	
	float *in1 = in1_d, *in2 = in2_d, *out1 = out1_d, *out2 = out2_d, *out = out_d, *outh = out_h;
	
	cudaStream_t stream_1, stream_2, stream_3;
	cudaStreamCreate(&stream_1);
	cudaStreamCreate(&stream_2);
	cudaStreamCreate(&stream_3);
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 dimBlock(thread_x, 1, 1);
	dim3 dimGrid(1, 1);
	
	cudaEventRecord(start, 0);
	
	process_1<<<dimGrid, dimBlock, 0, stream_1>>>(in1, threshold, out1);
	process_2<<<dimGrid, dimBlock, 0, stream_2>>>(in1, in2, out2);
	for(int i = 0; i < num_blocks-1; i++)
	{
		cudaStreamSynchronize(stream_1);
		cudaStreamSynchronize(stream_2);
		
		postprocess<<<dimGrid, dimBlock, 0, stream_3>>>(out1, out2, out);
		
		in1 += block_size;
		in2 += block_size;
		out1 += block_size;
		out2 += block_size;
		out += block_size; 
		
		process_1<<<dimGrid, dimBlock, 0, stream_1>>>(in1, threshold, out1);
		process_2<<<dimGrid, dimBlock, 0, stream_2>>>(in1, in2, out2);
	}
	cudaStreamSynchronize(stream_1);
	cudaStreamSynchronize(stream_2);
	postprocess<<<dimGrid, dimBlock, 0, stream_3>>>(out1, out2, out);
	cudaDeviceSynchronize();
	
	// copy back to cpu
	cudaError_t err = cudaMemcpy(out_h, out_d, num_blocks * block_size * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	 
	*elapsed_time = time; 
	
	cudaStreamDestroy(stream_1);
	cudaStreamDestroy(stream_2);
	cudaStreamDestroy(stream_3);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void compute_gpu_2(float* in1_d, float* in2_d, float threshold, float* out1_d, float* out2_d, float* out_d, float* out_h, float* elapsed_time)
{
	cudaEvent_t start, stop; 
	cudaEvent_t processing_done;
	
	float *in1 = in1_d, *in2 = in2_d, *out1 = out1_d, *out2 = out2_d, *out = out_d;
	float *outh = out_h;
	
	cudaStream_t stream_1, stream_2, stream_3, stream_4;
	cudaStreamCreate(&stream_1);
	cudaStreamCreate(&stream_2);
	cudaStreamCreate(&stream_3);
	cudaStreamCreate(&stream_4);
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&processing_done);
	
	dim3 dimBlock(thread_x, 1, 1);
	dim3 dimGrid(1, 1);
	
	cudaEventRecord(start, 0);
	
	process_1<<<dimGrid, dimBlock, 0, stream_1>>>(in1, threshold, out1);
	process_2<<<dimGrid, dimBlock, 0, stream_2>>>(in1, in2, out2);
	for(int i = 0; i < num_blocks-1; i++)
	{
		cudaStreamSynchronize(stream_1);
		cudaStreamSynchronize(stream_2);
		
		postprocess<<<dimGrid, dimBlock, 0, stream_3>>>(out1, out2, out);
		
		//while(cudaEventQuery(processing_done) != true);		// prevents from double recording
		cudaEventRecord(processing_done, stream_3);
		
		cudaStreamWaitEvent(stream_4, processing_done, 0);
		cudaMemcpyAsync(outh, out, block_size * sizeof(float), cudaMemcpyDeviceToHost, stream_4);
		
		in1 += block_size;
		in2 += block_size;
		out1 += block_size;
		out2 += block_size;
		out += block_size; 
		outh += block_size;
		
		process_1<<<dimGrid, dimBlock, 0, stream_1>>>(in1, threshold, out1);
		process_2<<<dimGrid, dimBlock, 0, stream_2>>>(in1, in2, out2);
	}
	cudaStreamSynchronize(stream_1);
	cudaStreamSynchronize(stream_2);
	postprocess<<<dimGrid, dimBlock, 0, stream_3>>>(out1, out2, out);
	
	//while(cudaEventQuery(processing_done) != true);		// prevents from double recording
	cudaEventRecord(processing_done, stream_3);
		
	cudaStreamWaitEvent(stream_4, processing_done, 0);
	cudaMemcpyAsync(outh, out, block_size * sizeof(float), cudaMemcpyDeviceToHost, stream_4);
	
	//cudaDeviceSynchronize(); 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	 
	*elapsed_time = time; 
	
	cudaStreamDestroy(stream_1);
	cudaStreamDestroy(stream_2);
	cudaStreamDestroy(stream_3);
	cudaStreamDestroy(stream_4);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(processing_done);
}
	
int main()
{
	float *in1_h, *in2_h, *out1_h, *out2_h, *out_h;
	float *in1_d, *in2_d, *out1_d, *out2_d, *out_d;
	
	float* out_gpu_1_h, *out_gpu_2_h;
	
	float threshold = (float)rand()/(float)RAND_MAX; 
	
	in1_h  = (float*)malloc(num_blocks * block_size * sizeof(float));
	in2_h  = (float*)malloc(num_blocks * block_size * sizeof(float));
	out1_h = (float*) malloc(num_blocks * block_size * sizeof(float));
	out2_h = (float*)malloc(num_blocks * block_size * sizeof(float));
	out_h  = (float*)malloc(num_blocks * block_size * sizeof(float));
	out_gpu_1_h= (float*)malloc(num_blocks * block_size * sizeof(float));
	
	cudaMallocHost((void**)&out_gpu_2_h, num_blocks * block_size * sizeof(float));
	
	cudaMalloc((void**)&in1_d, num_blocks * block_size * sizeof(float));
	cudaMalloc((void**)&in2_d, num_blocks * block_size * sizeof(float));
	cudaMalloc((void**)&out1_d, num_blocks * block_size * sizeof(float));
	cudaMalloc((void**)&out2_d, num_blocks * block_size * sizeof(float));
	cudaMalloc((void**)&out_d, num_blocks * block_size * sizeof(float));
	
	// generate input and do cpu computation
	for(int i = 0; i < num_blocks * block_size; i++)
	{
		in1_h[i] = (float)rand()/(float)RAND_MAX;
		in2_h[i] = (float)rand()/(float)RAND_MAX;
	}
	
	// compute on the cpu
	compute_cpu(in1_h, in2_h, threshold, out1_h, out2_h, out_h);
	
	// copy input to GPU
	cudaMemcpy(in1_d, in1_h, num_blocks * block_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(in2_d, in2_h, num_blocks * block_size * sizeof(float), cudaMemcpyHostToDevice);
	
	{
		// compute on the gpu ver 1
		float elapsed_time = 0.0;
		compute_gpu_1(in1_d, in2_d, threshold, out1_d, out2_d, out_d, out_gpu_1_h, &elapsed_time);
	
		// compute error
		float max_error = -1.0;
		int index = 0;
	
		float *ref = out_h;
		float *res = out_gpu_1_h;
		compute_error(ref, res, num_blocks * block_size, &max_error, &index);
	
		printf("Scenario 1: offload, compute, and copy back\n");
		printf("-------------------------------------------\n");
		printf("Elapsed time = %.2f ms\n", elapsed_time);
		printf("Maximum error = %.2f @ %d\n", max_error, index);
	}
	
	{
		// compute on the gpu ver 2
		float elapsed_time = 0.0;
		compute_gpu_2(in1_d, in2_d, threshold, out1_d, out2_d, out_d, out_gpu_2_h, &elapsed_time);

		// compute error
		float max_error = -1.0;
		int index = 0;
	
		float *ref = out_h;
		float *res = out_gpu_2_h;
		compute_error(ref, res, num_blocks * block_size, &max_error, &index);
	
		printf("\n\n");
		printf("Scenario 2: offload, concurrent compute and copy back\n");
		printf("-----------------------------------------------------\n");
		printf("Elapsed time = %.2f ms\n", elapsed_time);
		printf("Maximum error = %.2f @ %d\n", max_error, index);
	
	}
	
	
	if(in1_h) free(in1_h);
	if(in2_h) free(in2_h);
	if(out1_h)free(out1_h);
	if(out2_h)free(out2_h);
	if(out_h)free(out_h);
	if(out_gpu_1_h)free(out_gpu_1_h);
	
	if(out_gpu_2_h)cudaFreeHost(out_gpu_2_h);
	
	if(in1_d) cudaFree(in1_d);
	if(in2_d) cudaFree(in2_d);
	if(out1_d)cudaFree(out1_d);
	if(out2_d)cudaFree(out2_d);
	if(out_d)cudaFree(out_d);

	return 0;
}