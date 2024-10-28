#include <cuda_runtime.h>
#include <iostream>
#include "../ops.h"    

// CUDA Kernel for color image subtraction
__global__ void kernel(int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        printf("%d %d\n", col, row);
    }
    
}


int main()
{   
    int width = 10;
    int height = 10;
    dim3 threads(16, 16); 
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    std::cout << blocks.x << " " << blocks.y << std::endl;
    kernel<<<blocks, threads>>>(width, height);
    cudaDeviceSynchronize();
    return 0;
}
