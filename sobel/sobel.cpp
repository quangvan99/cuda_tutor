#include <opencv2/opencv.hpp>
#include <iostream>
#include "../ops.h"

void get_sobel(cv::Mat& grad, cv::Mat& grad_x, cv::Mat& grad_y, cv::Mat& abs_grad_x, cv::Mat& abs_grad_y, cv::Mat& img) {

    // Sobel operations
    int ddepth = CV_16S; // Use a higher depth to avoid overflow
    int ksize = 3; // Size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
    double scale = 1;
    double delta = 0;

    // Gradient X
    Sobel(img, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    // Convert back to CV_8U
    cv::convertScaleAbs(grad_x, abs_grad_x);

    // Gradient Y
    Sobel(img, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);
    // Convert back to CV_8U
    cv::convertScaleAbs(grad_y, abs_grad_y);

    // Total Gradient (approximate)
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);   
}

int main() {
    // Load the image in grayscale
    cv::Mat img = cv::imread("im/t1.jpg", cv::IMREAD_GRAYSCALE);

    cv::Size size(10000, 10000);
    cv::resize(img, img, size);

    cv::Mat grad;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    measure_exec_time(get_sobel, grad, grad_x, grad_y, abs_grad_x, abs_grad_y, img);
    cv::imwrite("sobel2cpp.jpg", grad);

    return 0;
}
