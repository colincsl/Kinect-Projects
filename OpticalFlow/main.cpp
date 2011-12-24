

//General dependencies

#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <time.h>
using namespace std;


// OpenCV depenedencies

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
using namespace cv;

// Globals


int main (int argc, char** argv)
{

    namedWindow("im1");
    namedWindow("im2");
    namedWindow("flowX");    
    namedWindow("flowY");    
    
    Mat img1 = imread("../particle test0606.png", 0);
    Mat img2 = imread("../particle test0607.png", 0);
    
    Mat flow(600, 800, CV_32FC2);
    Mat flowA(600, 800, CV_32FC1);    
    Mat flowB(600, 800, CV_32FC1);        
    
    // start, end, flowXY, pyrScale, #Pyrs, winSize, iters, NN, sigma, flags
    calcOpticalFlowFarneback(img1, img2, flow, .5, 5, 9, 9, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
    
    Point p1(1,1);
    Point p2(100,100);
    
    Mat flows[]  = {flowA, flowB};
    int fromTo[] = {0,0, 1,1};
    mixChannels(&flow, 1, flows, 2, fromTo, 2);    
    
    for (int y=0; y<img1.cols; y++)
    {
        for (int x=0; x<img1.rows; x++)
        {
            if (flows[0].at<float>(x,y) > 0)
            {
//                Point p1(x,y);
//                Point p2(x+flows[0].at<float>(x,y),y+flows[1].at<float>(x,y));
                Point p1(y,x);
                Point p2(y+flows[1].at<float>(x,y), x+flows[0].at<float>(x,y));
                line(img1, p1, p2, 255);
            }
        }
    }
    
    
    imshow("im1", img1);
//    imshow("im2", img2);
    
    imshow("flowX", flows[0]/100.0);
    imshow("flowY", flows[1]/100.0);    
    
    waitKey();
    
    
    
    
    return 0;
}
