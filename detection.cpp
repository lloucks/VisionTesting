#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <stdio.h>

// C++ adaptation of this code was done based off the tutorial from
// https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
// excecute with g++ detection.cpp -o detection `pkg-config --cflags --libs opencv`

#define MIN_MATCH_COUNT 10


int main(){
    cv::Mat jiangshi = cv::imread("images/Jiangshi_lg.png", cv::IMREAD_GRAYSCALE);
    cv::Mat test = cv::imread("images/test1.jpg", cv::IMREAD_GRAYSCALE);
    // step 1: Detect the keypoints using SURF Detector and compute the descriptors
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> sift = cv::xfeatures2d::SURF::create(minHessian);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(jiangshi, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(test, cv::noArray(), keypoints2, descriptors2);

    // Step 2: matching the descriptor vectors with a FLANN based matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2);

    // Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.6f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){
        if ( knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
        }
    }


    // Draw matches
    cv::Mat img_matches;
    // Checks if there are enough matches for it to say whether it detected the vampire
    // using https://docs.opencv.org/3.1.0/d7/dff/tutorial_feature_homography.html and
    // https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    if( good_matches.size() >= MIN_MATCH_COUNT){
        std::vector<cv::Point2f> src_pts;
        std::vector<cv::Point2f> dst_pts;
        // the below line draws the vectors showing the jiangshi feature matches
        cv::drawMatches( jiangshi, keypoints1, test, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        for( cv::DMatch x : good_matches){
            src_pts.push_back(keypoints1[x.queryIdx].pt);
            dst_pts.push_back(keypoints2[x.trainIdx].pt);
        }
        cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC);
        // get the corners from the jiangshi image
        std::vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( jiangshi.cols, 0 );
        obj_corners[2] = cvPoint( jiangshi.cols, jiangshi.rows ); obj_corners[3] = cvPoint( 0, jiangshi.rows );
        std::vector<cv::Point2f> scene_corners(4);
        
        perspectiveTransform( obj_corners, scene_corners, H);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + cv::Point2f( jiangshi.cols, 0), scene_corners[1] + cv::Point2f( jiangshi.cols, 0), cv::Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + cv::Point2f( jiangshi.cols, 0), scene_corners[2] + cv::Point2f( jiangshi.cols, 0), cv::Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + cv::Point2f( jiangshi.cols, 0), scene_corners[3] + cv::Point2f( jiangshi.cols, 0), cv::Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + cv::Point2f( jiangshi.cols, 0), scene_corners[0] + cv::Point2f( jiangshi.cols, 0), cv::Scalar( 0, 255, 0), 4 );

    }
    cv::resize(img_matches, img_matches, cv::Size(), 0.2, 0.2 );
    cv::imshow("Good Matches", img_matches);
    cv::waitKey();
    return 0;
}
