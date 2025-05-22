#include <iostream>
#include <cuda_runtime.h>

__global__ void sumArraysKernel(int *array, int *newArray, int numRows, int numCols) {
    int col = threadIdx.x;  // Thread index determines which column to sum
    
    if (col < numCols) {
        // Sum corresponding elements from each row for the given column
        newArray[2 * col] = array[2 * col] + array[2 * col + numCols];    // Summing first element pair
        newArray[2 * col + 1] = array[2 * col + 1] + array[2 * col + 1 + numCols];  // Summing second element pair
    }
}

/*array = [
  [  // Row 0
    [0, 1],  // Column 0: Sub-array [0, 1]
    [2, 3]   // Column 1: Sub-array [2, 3]
  ],
  [  // Row 1
    [4, 5],  // Column 0: Sub-array [4, 5]
    [6, 7]   // Column 1: Sub-array [6, 7]
  ]
]
FLATTEN BEFORE USAGE

array = [0, 1, 2, 3, 4, 5, 6, 7]
*/

int main() {
    const int numRows = 2;  // Number of rows (in this case, two arrays)
    const int numCols = 2;  // Number of columns (each array has two elements)
    
    // Input 2D array (flattened for CUDA)
    int array[numRows * numCols * 2] = {
        0, 1,  // First row: [0, 1]
        2, 3,  // First row: [2, 3]
        4, 5,  // Second row: [4, 5]
        6, 7   // Second row: [6, 7]
    };

    int newArray[numCols * 2];  // Resulting array to store the sum

    int *d_array, *d_newArray;

    // Allocate memory on the device
    cudaMalloc((void **)&d_array, numRows * numCols * 2 * sizeof(int));
    cudaMalloc((void **)&d_newArray, numCols * 2 * sizeof(int));

    // Copy the array to the device
    cudaMemcpy(d_array, array, numRows * numCols * 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with 1 block and numCols threads per block
    sumArraysKernel<<<1, numCols>>>(d_array, d_newArray, numRows, numCols);

    // Copy the result back to the host
    cudaMemcpy(newArray, d_newArray, numCols * 2 * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the result
    std::cout << "newArray = [";
    for (int i = 0; i < numCols; ++i) {
        std::cout << "[" << newArray[2 * i] << ", " << newArray[2 * i + 1] << "]";
        if (i < numCols - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_newArray);

    return 0;
}
