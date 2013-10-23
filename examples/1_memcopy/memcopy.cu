#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// CPU code
int main(int argc, char* argv[])
{
	if(argc < 2)
	{
		printf("USGAE: %s <array-length>\n", argv[0]);
		exit(-1);
	}
	
	int length = atoi(argv[1]);
	
	// cpu pointers
	float *ha, *hb;
	// gpu pointers
	float *da, *db;
	
	// allocate and initialize cpu memory
	ha = (float*)malloc(length * sizeof(float));
	hb = (float*)malloc(length * sizeof(float));
	for(int i = 0; i < length; i++) ha[i] = rand();		// init ha randomly
	memset(hb, 0, length * sizeof(float));				// init hb with zeros
	
	// allocate gpu memory
	cudaMalloc((void**)&da, length * sizeof(float));
	cudaMalloc((void**)&db, length * sizeof(float));
	
	printf("Copying from host to device .. ");
	// memory copy host to device
	cudaMemcpy(da, ha, length * sizeof(float), cudaMemcpyHostToDevice);
	printf("done\n\n");
	
	printf("Copying inside device memory .. ");
	// memory copy inside gpu memory
	cudaMemcpy(db, da, length * sizeof(float), cudaMemcpyDeviceToDevice);
	printf("done\n\n");
	
	printf("Copying back from device to host .. ");
	// memory copy from device (gpu) to host (cpu)
	cudaMemcpy(hb, db, length * sizeof(float), cudaMemcpyDeviceToHost);
	printf("done\n\n");
	
	
	printf("Checking gpu output .. ");
	
	// now ha and hb should be the same
	int cmp = memcmp(ha, hb, length * sizeof(float) );
	if(cmp == 0)printf("passed\n");
	else printf("failed at %d\n", cmp);
	
	// free cpu memory
	if(ha)free(ha);
	if(hb)free(hb);
	
	// free gpu memory
	if(da)cudaFree(da);
	if(db)cudaFree(db);
	return 0;	
}