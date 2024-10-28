#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "../ops.h" 

#define MASK_DIM 3
#define MASK_OFFSET 1

__constant__ float mask[MASK_DIM * MASK_DIM] = {
    1.0f, 0.0f, -1.0f,
    2.0f, 0.0f, -2.0f,
    1.0f, 0.0f, -1.0f};

__constant__ float mask2[MASK_DIM * MASK_DIM] = {
    1.0f, 2.0f, 1.0f,
    0.0f, 0.0f, 0.0f,
    -1.0f, -2.0f, -1.0f};

__global__ void simple_sobelXY(const unsigned char *srcImage, unsigned char *result, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col >= MASK_OFFSET) && (col < (width - MASK_OFFSET)) && (row >= MASK_OFFSET) && (row < (height - MASK_OFFSET)))
    {
        float Gx = 0.0f;
        float Gy = 0.0f;

        for (int ky = -MASK_OFFSET; ky <= MASK_OFFSET; ky++)
        {
            for (int kx = -MASK_OFFSET; kx <= MASK_OFFSET; kx++)
            {
                float fl = static_cast<float>(srcImage[(row + ky) * width + (col + kx)]);
                Gx += fl * mask[(ky + MASK_OFFSET) * MASK_DIM + (kx + MASK_OFFSET)];
                Gy += fl * mask2[(ky + MASK_OFFSET) * MASK_DIM + (kx + MASK_OFFSET)];
            }
        }

        float Gxy_abs = sqrtf(Gx * Gx + Gy * Gy);
        Gxy_abs = (Gxy_abs > 255) ? 255 : Gxy_abs;

        result[row * width + col] = static_cast<unsigned char>(Gxy_abs);
    }
}

void get_sobel(unsigned char *d_image, unsigned char *d_result, 
               unsigned char *h_result, cv::Mat& image,
                int width, int height, cudaStream_t &stream){
    
    dim3 threads(32, 32);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    cudaMemcpyAsync(d_image, image.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
    simple_sobelXY<<<blocks, threads, 0, stream>>>(d_image, d_result, width, height);
    cudaMemcpyAsync(h_result, d_result, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // Wait for stream operations to complete
}

int main()
{
    cv::Mat image = cv::imread("im/t1.jpg", cv::IMREAD_GRAYSCALE);

    cv::Size size(10000, 10000);
    cv::resize(image, image, size);

    int width = image.cols;
    int height = image.rows;

    unsigned char *d_image, *d_result;
    cudaMalloc(&d_image, width * height * sizeof(unsigned char));
    cudaMalloc(&d_result, width * height * sizeof(unsigned char));
    unsigned char *h_result = new unsigned char[width * height];
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    measure_exec_time(get_sobel, d_image, d_result, h_result, image, width, height, stream);

    cv::Mat result(height, width, CV_8U, h_result);
    cv::imwrite("sobelcu.jpg", result);

    cudaFree(d_image);
    cudaFree(d_result);
    delete[] h_result;
    cudaStreamDestroy(stream); // Destroy the stream
    return 0;
}