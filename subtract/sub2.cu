#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include "../ops.h"    

// CUDA Kernel for color image subtraction
__global__ void d_image_subtract(const unsigned char *d_img1, const unsigned char *d_img2, 
                                unsigned char *d_result, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = (row * width + col) * 3; // Index for 3-channel color image
        printf("%d %d %d\n", col, row, idx);
        for (int c = 0; c < 3; c++) { // Loop over color channels
            d_result[idx + c] = abs((int)d_img1[idx + c] - (int)d_img2[idx + c]);
        }
    }
}

void h_image_subtract(unsigned char *d_img1, unsigned char *d_img2, unsigned char *d_result, 
                     cv::Mat &h_img1, cv::Mat &h_img2, unsigned char *h_result, 
                     cudaStream_t &stream){
    int width = h_img1.cols;
    int height = h_img2.rows;
    size_t numBytes = width * height * 3 * sizeof(unsigned char); // Adjust for 3 channels
    cudaMemcpy(d_img1, h_img1.data, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, h_img2.data, numBytes, cudaMemcpyHostToDevice);                        
    dim3 threads(32, 32); // Smaller thread block size for better handling of larger data
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    d_image_subtract<<<blocks, threads, 0, stream>>>(d_img1, d_img2, d_result, width, height);

    cudaStreamSynchronize(stream); 
}

int main()
{
    // Load two color images
    cv::Mat h_img1 = cv::imread("im/t1.jpg", cv::IMREAD_COLOR);
    cv::Mat h_img2 = cv::imread("im/t2.jpg", cv::IMREAD_COLOR);

    // Resize images to the same size
    cv::Size newSize(10000, 10000); // Assuming a reasonable size
    cv::resize(h_img1, h_img1, newSize);
    cv::resize(h_img2, h_img2, newSize);

    int width = h_img1.cols;
    int height = h_img1.rows;
    size_t numBytes = width * height * 3 * sizeof(unsigned char); // Adjust for 3 channels

    unsigned char *d_img1, *d_img2, *d_result;
    cudaMalloc(&d_img1, numBytes);
    cudaMalloc(&d_img2, numBytes);
    cudaMalloc(&d_result, numBytes);
    unsigned char *h_result = new unsigned char[numBytes];

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    measure_exec_time(h_image_subtract, d_img1, d_img2, d_result, h_img1, h_img2, h_result, stream);
    cudaMemcpy(h_result, d_result, numBytes, cudaMemcpyDeviceToHost);
    cv::Mat result(height, width, CV_8UC3, h_result);
    cv::imwrite("../im/subcu2.jpg", result);

    // Clean up
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);
    cudaStreamDestroy(stream);

    return 0;
}
