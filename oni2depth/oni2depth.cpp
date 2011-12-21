/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *	
 * Author: Nico Blodow (blodow@cs.tum.edu)
 *         Radu Bogdan Rusu (rusu@willowgarage.com)
 *         Suat Gedikli (gedikli@willowgarage.com)
 *         Ethan Rublee (rublee@willowgarage.com)
 */

#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/common/time.h> //fps calculations
#include <pcl/io/oni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/range_image/range_image.h>
#include "pcl/visualization/range_image_visualizer.h"
#include <vector>
#include <string>
#include <pcl/common/time_trigger.h>

#include <stdio.h>
#include <stdlib.h>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace pcl;

#define SHOW_FPS 1
#if SHOW_FPS
#define FPS_CALC(_WHAT_) \
do \
{ \
static unsigned count = 0;\
static double last = pcl::getTime ();\
++count; \
if (pcl::getTime() - last >= 1.0) \
{ \
double now = pcl::getTime (); \
std::cout << "Average framerate("<< _WHAT_ << "): " << double(count)/double(now - last) << " Hz" <<  std::endl; \
count = 0; \
last = now; \
} \
}while(false)
#else
#define FPS_CALC(_WHAT_) \
do \
{ \
}while(false)
#endif

template <typename PointType>
class SimpleONIViewer
{
public:
    typedef pcl::PointCloud<PointType> Cloud;
    typedef typename Cloud::ConstPtr CloudConstPtr;    
    bool lock;
    float minRange_, maxRange_;
    float angRez;
    float noiseLevel;
    float min_camera_range;
    int border_size;
    int i_image;
    pcl::RangeImage::CoordinateFrame coordinateFrame;
    Eigen::Affine3f sceneSensorPose;
    
    pcl::RangeImage rangeImage;
    pcl::visualization::RangeImageVisualizer rangeImageWidget;  
    Mat cvImg;
    
    
    SimpleONIViewer(pcl::ONIGrabber& grabber, float minRange=-1.0f, float maxRange=-1.0f)
    : grabber_(grabber), lock(false), minRange_(minRange), maxRange_(maxRange), cvImg(Size(640, 480), CV_8U)
    {
//        cvImg(Size(640, 480), CV_16U)
//        viewer("PCL OpenNI Viewer")
        angRez = pcl::deg2rad(0.09f);
        noiseLevel = 0.0f;
        min_camera_range = 0.0f;
        border_size = 0;
        i_image = 0;
        coordinateFrame = pcl::RangeImage::CAMERA_FRAME;
        sceneSensorPose = Eigen::Affine3f::Identity();
    }
    
    /**
     * @brief Callback method for the grabber interface
     * @param cloud The new point cloud from Grabber
     */
    void
    cloud_cb_ (const CloudConstPtr& cloud)
    {

        FPS_CALC ("callback");
        boost::mutex::scoped_lock lock (mtx_);
        cloud_ = cloud;
        
        rangeImage.createFromPointCloud(*cloud, angRez, pcl::deg2rad(360.0f), pcl::deg2rad(180.0f), sceneSensorPose, coordinateFrame, noiseLevel, min_camera_range, border_size);
        
        
        if (minRange_ < 0.0f)
        {
            rangeImage.getMinMaxRanges(minRange_, maxRange_);                       
            std::cout << "Min: " << minRange_ << " Max: " << maxRange_ << std::endl;
        }    

        rangeImage.setUnseenToMaxRange();
     //   minRange_ = 1.5;
       // maxRange_ = 2.8;

        
//        float rangeFloats[640*480];
//        float *rangeFloats;
//        rangeFloats = rangeImage.getRangesArray();
        
        std::string img_base("output/depth_");        
        std::string img_filename;        
        std::string ind_string;
        char i_image_char[10];
        
        sprintf(i_image_char, "%04d", i_image);
        img_filename = img_base + i_image_char + ".jpg";                                  
        
        std::cout << img_filename << std::endl;

        
//        MatIterator_<uchar> it = cvImg.begin();
////        it = cvImg.begin<uchar>();
//        MatIterator_<uchar> it_end = cvImg.end<uchar>();
//        for(; it != it_end; ++it)
//        {
//            cout << it << endl;
////            double v = *it * 1.7 + rand()
//            double v = rangeFloats[*it];
//            *it = saturate_cast<uchar>(v/maxRange_);
//        }


        try {

            float tmp_z;
            PointWithRange tmp_pt;
            char rgb_val;
            short short_val;
            float maxDist = minRange_+maxRange_;
            unsigned int iter = 0;
            for (int y=0; y<480; y++)        
            {
                for (int x=0; x<640; x++)
                {
                    tmp_pt = rangeImage.getPoint(x, y);
                    tmp_z = tmp_pt.range;
                    
                    
                    if (tmp_z > maxRange_) tmp_z = maxRange_;                    
//                    cout << tmp_z << endl;
                    
                    
                    rgb_val = saturate_cast<char>((tmp_z-minRange_)/(maxRange_-minRange_)*255.);
//                    short_val = saturate_cast<unsigned short>((tmp_z-minRange_)/(maxRange_-minRange_)*65535);                    
                    cvImg.at<char>(y, x) = rgb_val;
//                    cvImg.at<unsigned short>(y, x) = short_val; 
//                    cout << cvImg.at<char>(y, x) << endl;
//                    cout << cvImg.at<unsigned short>(y, x) << endl;                    
//                    cout << endl;
                    iter++;
                }
            }
            
            imwrite(img_filename, cvImg);
            
            i_image++; // increment image number

            
        } catch (char ee) {}
            
//        delete rangeFloats;
        
    }
    
    /**
     * @brief swaps the pointer to the point cloud with Null pointer and returns the cloud pointer
     * @return boost shared pointer to point cloud
     */
//    CloudConstPtr
//    getLatestCloud ()
//    {
//        //lock while we swap our cloud and reset it.
//        boost::mutex::scoped_lock lock(mtx_);
//        CloudConstPtr temp_cloud;
//        temp_cloud.swap (cloud_); //here we set cloud_ to null, so that
//        //it is safe to set it again from our
//        //callback
//        return (temp_cloud);
//    }
    
    /**
     * @brief starts the main loop
     */
    void
    run()
    {
        boost::function<void (const CloudConstPtr&) > f = boost::bind (&SimpleONIViewer::cloud_cb_, this, _1);        
        boost::signals2::connection c = grabber_.registerCallback (f);
        
        grabber_.start();
        
        while (1) {
            boost::this_thread::sleep(boost::posix_time::seconds(1));
        }
        
        grabber_.stop();
    }
    
//    pcl::visualization::CloudViewer viewer;
    pcl::ONIGrabber& grabber_;
    boost::mutex mtx_;
    CloudConstPtr cloud_;
};

void
usage(char ** argv)
{
    cout << "usage: " << argv[0] << " <path-to-oni-file> [min range] [max range] \n";
    cout << argv[0] << " -h | --help : shows this help" << endl;
    return;
}

int
main(int argc, char ** argv)
{
    std::string arg("");
    
    float minRange_in = -1.0f;
    float maxRange_in = -1.0f;
    
    unsigned frame_rate = 0;
    if (argc >= 2)
    {
        arg = argv[1]; // Filename
        
        if (arg == "--help" || arg == "-h")
        {
            usage(argv);
            return 1;
        }
        
        if (argc >= 4)
        {
            minRange_in = atoi(argv[2]);
            maxRange_in = atoi(argv[3]);            
        }
    }
    else
    {
        usage (argv);
        return 1;
    }
    
    pcl::TimeTrigger trigger;
    
    pcl::ONIGrabber* grabber = 0;
    if (frame_rate == 0)
        grabber = new  pcl::ONIGrabber(arg, true, true);
    else
    {
        grabber = new  pcl::ONIGrabber(arg, true, false);
//        trigger.setInterval (1.0 / (double) frame_rate);
//        trigger.registerCallback (boost::bind(&pcl::ONIGrabber::start, grabber));
//        trigger.start();
    }
    if (grabber->providesCallback<pcl::ONIGrabber::sig_cb_openni_point_cloud_rgb > ())
    {
        SimpleONIViewer<pcl::PointXYZRGB> v(*grabber, minRange_in, maxRange_in);
        v.run();
    }
    else
    {
        SimpleONIViewer<pcl::PointXYZI> v(*grabber, minRange_in, maxRange_in);
        v.run();
    }
    
    return (0);
}
