#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <stdio.h>

// excecute with g++ detection.cpp -o detection `pkg-config --cflags --libs opencv`

int main(){
    auto jiangshi = cv::imread("images/Jiangshi_lg.png");
    std::cout << "Width : " << jiangshi.cols << std::endl;
    sift = cv::xfeatures2d
    // resize the image for viewing from the computer
}