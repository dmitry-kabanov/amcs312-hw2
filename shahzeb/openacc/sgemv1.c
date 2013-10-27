#include <stdio.h>
#include "timer.h"
#include <stdlib.h>
#include <math.h>
#include <openacc.h>

//#define DEBUG


int main(int argc, char *argv[])
{
  float *a,*b,* restrict c;


  int devnum; 
  acc_device_t acc_device_nvidia;
  acc_set_device_num(1,acc_device_nvidia);
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

  // Initialize matrices.
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
     a[i*size+j] = (float)rand()/(float)RAND_MAX;
    }
    b[i] = (float)rand()/(float)RAND_MAX;
    c[i] = 0;
  }
  double runtime;
  StartTimer();
  // Compute matrix vector multiplication.
  #pragma acc kernels copy(a[0:size*size],b[0:size],c[0:size])
  {
  #pragma acc loop independent gang vector 
  for (i = 0; i < size; ++i) {
    float sum = 0.f;
    #pragma acc loop vector
    for (j = 0; j < size; ++j) {
	sum += a[i*size+j] * b[j];
      }
      c[i] = sum;
    }
  }
 
  runtime = GetTimer();
  float sizef = (float)size;
  double TotalMFLOPS = 2*1E-6 * pow(sizef,2) + 2*sizef;
  printf("execution time: %f s\n", runtime / 1000.f);
  printf("MFLOPS/sec: %f\n",TotalMFLOPS/(runtime/1000.f)); 
#ifdef DEBUG
  for (i = 0; i < size; i++)
  {
    for (j = 0; j < size; j++)
    {
	printf("%0.2f ",a[i*size+j]);
    }
    printf("\t %0.2f \t %0.2f\n ",b[i],c[i]);
  }
#endif
  return 0;
}
