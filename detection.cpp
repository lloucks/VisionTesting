#include <stdio.h>

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "buoyDetector.cpp"


// C++ adaptation of this code was done based off the tutorial from
// https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
// compile with g++ detection.cpp -o detection `pkg-config --cflags --libs opencv`

#define MIN_MATCH_COUNT 10


int main(){ 

    // capture video from webcam
    cv::VideoCapture cap(0); 
    BuoyDetector detector(Jiangshi, cap);
    while(1){
        detector.Demo();
        if(cv::waitKey(30) >= 0) break;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}