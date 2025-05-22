#include <iostream>
#include <cuda_runtime.h>

__global__ void sumRowsKernel(int *array, int *newArray, int numRows, int numCols) {
    int row = blockIdx.x;  // Use blockIdx.x to index the row

    if (row < numRows) {
        int sum = 0;
        for (int col = 0; col < numCols; col++) {
            sum += array[row * numCols + col];  // Access the array element at (row, col)
        }
        newArray[row] = sum;  // Store the sum in the newArray
    }
}

int main() {
    const int numRows = 2;  // Number of rows in the array
    const int numCols = 2;  // Number of columns in the array
    int array[numRows][numCols] = { {0, 1}, {2, 3} };  // Original 2D array
    int newArray[numRows];  // Array to store results

    int *d_array, *d_newArray;

    // Allocate memory on the device
    cudaMalloc((void **)&d_array, numRows * numCols * sizeof(int));
    cudaMalloc((void **)&d_newArray, numRows * sizeof(int));

    // Copy the array to the device
    cudaMemcpy(d_array, array, numRows * numCols * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with numRows blocks and 1 thread per block
    sumRowsKernel<<<numRows, 1>>>(d_array, d_newArray, numRows, numCols);

    // Copy the result back to the host
    cudaMemcpy(newArray, d_newArray, numRows * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the result
    std::cout << "newArray = [";
    for (int i = 0; i < numRows; ++i) {
        std::cout << newArray[i];
        if (i < numRows - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_newArray);

    return 0;
}
