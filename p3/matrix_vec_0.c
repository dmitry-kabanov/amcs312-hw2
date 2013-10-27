#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matVecKernel(float **a, float *x, float *restrict y, size_t dim)
{
    size_t i;
    size_t k;
    for (i = 0; i < dim; ++i) {
        for (k = 0; k < dim; ++k) {
            y[i] += a[i][k] * x[k];
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("USAGE: ./matrix.exe <vector-dimension>\n\n");
        exit(1);
    }

    size_t dim = atoi(argv[1]);

    size_t i;
    size_t j;
    size_t k;
    
    time_t start_accel;
    time_t stop_accel;
    double time_accel;

    time_t start_cpu;
    time_t stop_cpu;
    double time_cpu;

    float **a = (float **) malloc(dim * sizeof(float *));
    float *x = (float *) malloc(dim * sizeof(float));
    float *y = (float *) malloc(dim * sizeof(float));
    float *y_check = (float *) malloc(dim * sizeof(float));

    for (i = 0; i < dim; ++i) {
        a[i] = (float *) malloc(dim * sizeof(float));
    }

    for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
            a[i][j] = (float) i + j;
        }
        x[i] = i;
        y[i] = 0.0f;
        y_check[i] = 0.0f;
    }

    start_accel = clock();
    matVecKernel(a, x, y, dim);
    stop_accel = clock();

    start_cpu = clock();
    for (i = 0; i < dim; ++i) {
        for (k = 0; k < dim; ++k) {
            y_check[i] += a[i][k] * x[k];
        }
    }
    stop_cpu = clock();

    /* Check */
    for (i = 0; i < dim; ++i) {
        if (y[i] != y_check[i]) {
            printf("Element [%d] is not the same.\n", i, j);
            exit(1);
        }
    }

    printf("Vectors are equal.\n");

    time_accel = ((double) stop_accel - start_accel) / CLOCKS_PER_SEC;
    time_cpu = ((double) stop_cpu - start_cpu) / CLOCKS_PER_SEC;

    printf("Time on accelerator: %g seconds\n", time_accel);
    printf("FLOPS on accelerator: %f GFlops/sec\n", 2 * dim * dim * 1.0e-9 / time_accel);
    printf("Time on CPU: %g seconds\n", time_cpu);
    printf("FLOPS on CPU: %f GFlops/sec\n", 2 * dim * dim * 1.0e-9 / time_cpu);
    
    return 0;
}
