#include <cuda_runtime.h>
#include <stdio.h>


// CUDA kernel for 2D convolution
__global__ void convolution2D(
    const float* input,
    float* output,
    const float* kernel,
    int imageWidth,
    int imageHeight,
    int kernelSize
) {
    // Calculate thread indices
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if thread is within image bounds
    if (tx < imageWidth && ty < imageHeight) {
        float sum = 0.0f;
        int kernelRadius = kernelSize / 2;
        
        // Perform convolution
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                int imgX = tx + kx;
                int imgY = ty + ky;
                
                // Handle boundary conditions (zero padding)
                if (imgX >= 0 && imgX < imageWidth && imgY >= 0 && imgY < imageHeight) {
                    float pixelValue = input[imgY * imageWidth + imgX];
                    float kernelValue = kernel[(ky + kernelRadius) * kernelSize + 
                                             (kx + kernelRadius)];
                    sum += pixelValue * kernelValue;
                }
            }
        }
        
        // Write output
        output[ty * imageWidth + tx] = sum;
    }
}

// Host function to perform convolution
void performConvolution(
    const float* h_input,
    float* h_output,
    const float* h_kernel,
    int imageWidth,
    int imageHeight,
    int kernelSize
) {
    // Allocate device memory
    float *d_input, *d_output, *d_kernel;
    size_t imageSize = imageWidth * imageHeight * sizeof(float);
    size_t kernelBytes = kernelSize * kernelSize * sizeof(float);
    
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelBytes);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (imageWidth + blockDim.x - 1) / blockDim.x,
        (imageHeight + blockDim.y - 1) / blockDim.y
    );
    
    // Launch kernel
    convolution2D<<<gridDim, blockDim>>>(
        d_input, d_output, d_kernel,
        imageWidth, imageHeight, kernelSize
    );
    
    // Check for kernel launch errors
    cudaGetLastError();
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}


void printMatrix(const float* matrix, int width, int height, const char* label) {
    printf("\n%s:\n", label);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%.2f ", matrix[y * width + x]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Test image dimensions
    const int imageWidth = 6;
    const int imageHeight = 6;
    const int kernelSize = 3;

    // Create test input image (6x6)
    float h_input[imageWidth * imageHeight] = {
        1, 1, 1, 1, 1, 1,
        1, 2, 2, 2, 2, 1,
        1, 2, 3, 3, 2, 1,
        1, 2, 3, 3, 2, 1,
        1, 2, 2, 2, 2, 1,
        1, 1, 1, 1, 1, 1
    };

    // Create test kernel (3x3 Gaussian blur-like)
    float h_kernel[kernelSize * kernelSize] = {
        1.0f/16, 2.0f/16, 1.0f/16,
        2.0f/16, 4.0f/16, 2.0f/16,
        1.0f/16, 2.0f/16, 1.0f/16
    };

    // Allocate output array
    float h_output[imageWidth * imageHeight] = {0};

    // Print input data
    printMatrix(h_input, imageWidth, imageHeight, "Input Image");
    printMatrix(h_kernel, kernelSize, kernelSize, "Convolution Kernel");

    // Perform convolution
    performConvolution(
        h_input,
        h_output,
        h_kernel,
        imageWidth,
        imageHeight,
        kernelSize
    );

    // Print result
    printMatrix(h_output, imageWidth, imageHeight, "Output Image");

    return 0;
}
