#include "opencv2/opencv.hpp"
#include "opencv2/cudaarithm.hpp"
#include "../ops.h"

void cuda_opencv_subtract(cv::cuda::GpuMat &d_img1, cv::cuda::GpuMat &d_img2, cv::cuda::GpuMat &d_result, 
                          cv::Mat &h_img1, cv::Mat &h_img2, cv::Mat &result)
{
    d_img1.upload(h_img1);
    d_img2.upload(h_img2);
    cv::cuda::subtract(d_img1, d_img2, d_result);
    d_result.download(result);
}

int main()
{
    cv::Mat image1 = cv::imread("im/t1.jpg");
    cv::Mat image2 = cv::imread("im/t2.jpg");


    // Resize images to ensure they are the same size
    cv::Size size(10000, 10000); // Example size, adjust as needed
    cv::resize(image1, image1, size);
    cv::resize(image2, image2, size);

    // Upload images to the GPU
    cv::cuda::GpuMat d_img1, d_img2, d_result;
    cv::Mat result;

    measure_exec_time(cuda_opencv_subtract, d_img1, d_img2, d_result, image1, image2, result);

    cv::imwrite("../im/sub_cucv.jpg", result); // Save the result

    return 0;
}
