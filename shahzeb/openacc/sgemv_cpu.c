#include <stdio.h>
#include "timer.h"
#include <stdlib.h>
#include <math.h>

//#define DEBUG


int main(int argc, char *argv[])
{
  float *a,*b,*c;
  if (argc < 2)
  {
    printf("USAGE: <matrix-size>");
    exit(1);
  }
  
  int i,j,k;
  int size = atoi(argv[1]);

  double runtime;
  StartTimer();

  printf("allocating...\n");  
  a = (float*)malloc(sizeof(float*)*size*size);
  b = (float*)malloc(sizeof(float)*size);
  c = (float*)malloc(sizeof(float)*size);
  printf("initializing...\n");
  
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      a[i*size+j] = (float)rand()/(float)RAND_MAX;
    }
    b[i] = (float)rand()/(float)RAND_MAX;
    c[i] = 0;
  }
  for (i = 0; i < size; ++i) {
    float sum = 0;
    for (j = 0; j < size; ++j) {
	sum += a[i*size+j] * b[j];
      }
      c[i] = sum;
    }

  runtime = GetTimer();
  double TotalMFLOPS = 2*1E-6*pow(size,2) + 2*size;
  printf("execution time: %f s\n", runtime / 1000.f);
  printf("MFLOPS/sec: %f\n",TotalMFLOPS/(runtime/1000.f)); 
#ifdef DEBUG
  for (i = 0; i < size; i++)
  {
    for (j = 0; j < size; j++)
	printf("%0.2f ",a[i*size+j]);
    printf("\t %0.2f \t %0.2f\n ",b[i],c[i]);
  }
#endif
  return 0;
}
