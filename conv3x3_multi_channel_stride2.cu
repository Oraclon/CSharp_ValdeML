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
