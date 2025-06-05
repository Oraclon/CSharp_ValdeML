__global__ void conv3x3_multi_channel_stride2(
    const float* input,      // C * H * W
    const float* kernels,    // C * 3 * 3
    float* output,           // H_out * W_out
    int in_width,
    int in_height,
    int channels)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int kernel_size = 3;
    int stride = 2;

    int out_width = (in_width - kernel_size) / stride + 1;
    int out_height = (in_height - kernel_size) / stride + 1;

    if (out_x < out_width && out_y < out_height) {
        float sum = 0.0f;

        for (int c = 0; c < channels; ++c) {
            const float* channel_input = input + c * in_width * in_height;
            const float* kernel = kernels + c * kernel_size * kernel_size;

            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_x = out_x * stride + kx;
                    int in_y = out_y * stride + ky;

                    float val = channel_input[in_y * in_width + in_x];
                    float k = kernel[ky * kernel_size + kx];
                    sum += val * k;
                }
            }
        }

        output[out_y * out_width + out_x] = sum;
    }
}

int main() {
    const int channels = 3;
    const int in_width = 6;
    const int in_height = 6;
    const int kernel_size = 3;
    const int stride = 2;
    const int out_width = (in_width - kernel_size) / stride + 1;
    const int out_height = (in_height - kernel_size) / stride + 1;

    const int input_size = channels * in_width * in_height;
    const int kernel_size_total = channels * kernel_size * kernel_size;
    const int output_size = out_width * out_height;

    // Allocate and initialize host input
    float h_input[input_size];
    for (int i = 0; i < input_size; ++i) {
        h_input[i] = static_cast<float>(i + 1);  // Fill with 1, 2, 3, ...
    }

    // Simple kernels: one per channel
    float h_kernels[kernel_size_total] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1,

        0, 1, 0,
        0, 1, 0,
        0, 1, 0,

        -1, -1, -1,
         0,  0,  0,
         1,  1,  1
    };

    float h_output[output_size] = {0};

    // Allocate device memory
    float *d_input, *d_kernels, *d_output;
    cudaMalloc(&d_input, sizeof(float) * input_size);
    cudaMalloc(&d_kernels, sizeof(float) * kernel_size_total);
    cudaMalloc(&d_output, sizeof(float) * output_size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, sizeof(float) * input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, h_kernels, sizeof(float) * kernel_size_total, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(8, 8);
    dim3 gridSize((out_width + blockSize.x - 1) / blockSize.x,
                  (out_height + blockSize.y - 1) / blockSize.y);

    conv3x3_multi_channel_stride2<<<gridSize, blockSize>>>(
        d_input, d_kernels, d_output, in_width, in_height, channels);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_output, d_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost);

    // Print output
    std::cout << "Output:\n";
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            std::cout << h_output[y * out_width + x] << "\t";
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernels);
    cudaFree(d_output);

    return 0;
}
