#include <stdio.h>
#include "timer.h"
#include <stdlib.h>
#include <math.h>
#include <openacc.h>


int main(int argc, char *argv[])
{
  float *a,*b,*c;


  int devnum; 
  acc_device_t acc_device_nvidia;
  acc_set_device_num(2,acc_device_nvidia);
  // Initialize openacc runtime 
  acc_init(acc_device_nvidia);
  // Verification
  devnum = acc_get_device_num(acc_device_nvidia);
  printf(" I offload on device number:%d\n",devnum);

  if (argc < 2)
  {
    printf("USAGE: <matrix-size>");
    exit(1);
  }
  
  int i,j,k;
  int size = atoi(argv[1]);
  printf("allocating...\n");  
  a = (float*)malloc(sizeof(float)*size*size);
  b = (float *)malloc(sizeof(float)*size);
  c = (float *)malloc(sizeof(float)*size);

 
  printf("initializing...\n");
  int gangarr[100], vectorarr[32];
  for (i = 0; i < 100; i++)
    gangarr[i] = 32 + (i+1)*4;
  for (i = 0; i < 32; i++)
    vectorarr[i] = (i+1)*32;

  // Initialize matrices.
//  #pragma acc data copy(a[0:size][0:size],b[0:size],c[0:size])
//  {
//  #pragma acc kernels copyout(a[0:size*size],b[0:size],c[0:size])
//  {
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
     a[i*size+j] = (float)rand()/(float)RAND_MAX;
    }
    b[i] = (float)rand()/(float)RAND_MAX;
    c[i] = 0;
  }
  
//  printf("Complete Kernel\n");
  double runtime,mintime = 10E5;
  int bestgang, bestvector;
  int cnt = 1, gangindex = 0, vectorindex = 0;
  printf("Gang \t Vector \t Time\n");
  while (cnt < 100*32)
  {
  StartTimer();
  // Compute matrix vector multiplication.
  //#pragma acc data copyin(a[0:size][0:size],b[0:size]) copy(c[0:size])
  //{
  #pragma acc kernels copyin(a[0:size*size],b[0:size]) copy(c[0:size]) 
  #pragma acc loop gang(gangarr[gangindex]) vector(vectorarr[vectorindex]) independent
  for (i = 0; i < size; ++i) {
    float sum = 0;
    #pragma acc loop vector
    for (j = 0; j < size; ++j) {
	#pragma acc cache(b[0:size])
	sum += a[i*size+j] * b[j];
      }
      c[i] = sum;
    }
  runtime = GetTimer();
  memset(c,0,sizeof(float)*size);
  printf("%d \t %d \t %0.5e\n",gangarr[gangindex],vectorarr[vectorindex],runtime/1000.f);
  if (runtime < mintime)
  {
    bestgang = gangarr[gangindex];
    bestvector = vectorarr[vectorindex];
    mintime = runtime;
  }
  vectorindex++;
  if (cnt % 32 == 0)
  {
    gangindex++;
    vectorindex = 0;
  }
  cnt++;
  }
  printf("Best Configuration gang: %d \t vector: %d\t time: %0.5e\n",bestgang,bestvector,mintime);
  return 0;
}
