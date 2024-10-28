#include <iostream>
#include <opencv2/cudafilters.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "../ops.h"

void get_sobel(cv::cuda::GpuMat& d_resultx, cv::cuda::GpuMat& d_resulty, cv::cuda::GpuMat& d_resultxy, 
    cv::cuda::GpuMat& d_img, cv::Mat& img, cv::Mat& result){
    cv::Ptr<cv::cuda::Filter> filterx, filtery;
    filterx = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 0);
    filtery = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 0, 1);

    d_img.upload(img);
    filterx->apply(d_img, d_resultx);
    filtery->apply(d_img, d_resulty);

    cv::cuda::add(d_resultx, d_resulty, d_resultxy);

    d_resultxy.download(result);
}

int main()
{
    cv::Mat img = cv::imread("im/t1.jpg", cv::IMREAD_GRAYSCALE);

    cv::Size size(10000, 10000);
    cv::resize(img, img, size);

    cv::cuda::GpuMat d_img, d_resultx, d_resulty, d_resultxy;
    cv::Mat result;
    measure_exec_time(get_sobel, d_resultx, d_resulty, d_resultxy, d_img, img, result);

    cv::imwrite("sobelcpp.jpg", result);

    return 0;
}