#include <iostream>
#include <opencv2/opencv.hpp>
#include "../ops.h"

void subtract(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& res) {
    cv::subtract(img1, img2, res);
}

int main() {

    //  Get Build Information()
    std::cout << cv::getBuildInformation() << std::endl;

    cv::Mat image1 = cv::imread("im/t1.jpg");
    cv::Mat image2 = cv::imread("im/t2.jpg");

    cv::Size newSize(10000, 10000); 
    cv::resize(image1, image1, newSize);
    cv::resize(image2, image2, newSize);

    cv::Mat result;
    measure_exec_time(subtract, image1, image2, result);
    cv::imwrite("../im/subcpp.jpg", result); // Save the result
    return 0;
}