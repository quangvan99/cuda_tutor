#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include "../ops.h"    

// CUDA Kernel for color image subtraction with loop unrolling and using shared memory
__global__ void d_image_subtract(const unsigned char *d_img1, const unsigned char *d_img2, 
                                 unsigned char *d_result, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (row * width + col) * 3;

    if (col < width && row < height) {
        d_result[idx] = abs((int)d_img1[idx] - (int)d_img2[idx]);
        d_result[idx + 1] = abs((int)d_img1[idx + 1] - (int)d_img2[idx + 1]);
        d_result[idx + 2] = abs((int)d_img1[idx + 2] - (int)d_img2[idx + 2]);
    }
}

void h_image_subtract(unsigned char *d_img1, unsigned char *d_img2, unsigned char *d_result, 
                      cv::Mat &h_img1, cv::Mat &h_img2, unsigned char *h_result, 
                      cudaStream_t &stream) {
    int width = h_img1.cols;
    int height = h_img2.rows;
    size_t numBytes = width * height * 3 * sizeof(unsigned char); // Adjust for 3 channels
    cudaMemcpyAsync(d_img1, h_img1.data, numBytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_img2, h_img2.data, numBytes, cudaMemcpyHostToDevice, stream);
    dim3 threads(32, 32); // Smaller thread block size for better handling of larger data
    dim3 blocks(ceil(width / (float)threads.x), ceil(height / (float)threads.y));
    d_image_subtract<<<blocks, threads, 0, stream>>>(d_img1, d_img2, d_result, width, height);
    cudaMemcpyAsync(h_result, d_result, numBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

int main() {
    // Load two color images
    cv::Mat h_img1 = cv::imread("im/t1.jpg", cv::IMREAD_COLOR);
    cv::Mat h_img2 = cv::imread("im/t2.jpg", cv::IMREAD_COLOR);

    // Resize images to the same size
    cv::Size newSize(2000, 2000); // Assuming a smaller and more practical size
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

    cv::Mat result(height, width, CV_8UC3, h_result);
    cv::imwrite("sub7.jpg", result);

    // Clean up
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_result);
    cudaStreamDestroy(stream);
    delete[] h_result;

    return 0;
}
