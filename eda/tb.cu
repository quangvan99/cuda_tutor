#include <iostream>

// CUDA Kernel for color image subtraction
__global__ void kernel(int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        printf("x: %d, y: %d\n", x, y);
    }
    
}

int main()
{   
    int width = 4;
    int height = 5;

    dim3 threads(16, 16);
    dim3 blocks(ceil(width / (float)threads.x), ceil(height / (float)threads.y)); 
    std::cout << blocks.x << " " << threads.x << " " << blocks.y << " " << threads.y << std::endl;
    kernel<<<blocks, threads>>>(width, height);
    cudaDeviceSynchronize();
    return 0;
}
