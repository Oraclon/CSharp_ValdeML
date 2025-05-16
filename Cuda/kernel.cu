extern "C" __global__ void addOne(float** data, int rows, int cols)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        data[y][x] += 1.0f;
    }
}
