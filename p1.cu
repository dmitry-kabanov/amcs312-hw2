#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Memory clock rate: %d kHz\n", prop.memoryClockRate);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);

    return 0;
}
