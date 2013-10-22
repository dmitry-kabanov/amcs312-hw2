int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Clock rate: %d\n", prop.clockRate);

    return 0;
}
