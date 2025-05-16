#include <cuda_runtime.h>
#include <iostream>

extern void addOne(float** data, int rows, int cols);

extern "C" __declspec(dllexport) void LaunchKernel(float* hostData, int rows, int cols)
{
    float** device2D;
    float* deviceData;

    size_t totalSize = rows * cols * sizeof(float);
    cudaMalloc(&deviceData, totalSize);
    cudaMemcpy(deviceData, hostData, totalSize, cudaMemcpyHostToDevice);

    // Allocate row pointers
    float** hostPointers = new float*[rows];
    for (int i = 0; i < rows; ++i)
        hostPointers[i] = deviceData + i * cols;

    cudaMalloc(&device2D, rows * sizeof(float*));
    cudaMemcpy(device2D, hostPointers, rows * sizeof(float*), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    addOne<<<grid, block>>>(device2D, rows, cols);

    cudaMemcpy(hostData, deviceData, totalSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceData);
    cudaFree(device2D);
    delete[] hostPointers;
}
