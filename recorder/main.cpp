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
    float minRange_;
    float maxRange_;
    
    SimpleONIViewer(pcl::ONIGrabber& grabber, float minRange=-1.0f, float maxRange=-1.0f)
    : viewer("PCL OpenNI Viewer")
    , grabber_(grabber), lock(false), minRange_(minRange), maxRange_(maxRange)
    {
    }
    
    /**
     * @brief Callback method for the grabber interface
     * @param cloud The new point cloud from Grabber
     */
    void
    cloud_cb_ (const CloudConstPtr& cloud)
    {
//        if (!lock){
//                                std::cout << "Start new cloud" << std::endl;
            FPS_CALC ("callback");
            boost::mutex::scoped_lock lock (mtx_);
            cloud_ = cloud;
//                                std::cout << "End new cloud" << std::endl;
//        }
        
    }
    
    /**
     * @brief swaps the pointer to the point cloud with Null pointer and returns the cloud pointer
     * @return boost shared pointer to point cloud
     */
    CloudConstPtr
    getLatestCloud ()
    {
        //lock while we swap our cloud and reset it.
        boost::mutex::scoped_lock lock(mtx_);
        CloudConstPtr temp_cloud;
        temp_cloud.swap (cloud_); //here we set cloud_ to null, so that
        //it is safe to set it again from our
        //callback
        return (temp_cloud);
    }
    
    /**
     * @brief starts the main loop
     */
    void
    run()
    {
        //pcl::Grabber* interface = new pcl::OpenNIGrabber(device_id_, pcl::OpenNIGrabber::OpenNI_QQVGA_30Hz, pcl::OpenNIGrabber::OpenNI_VGA_30Hz);
        
        boost::function<void (const CloudConstPtr&) > f = boost::bind (&SimpleONIViewer::cloud_cb_, this, _1);
        
        boost::signals2::connection c = grabber_.registerCallback (f);
        
        grabber_.start();
        
        pcl::RangeImage rangeImage;
        pcl::visualization::RangeImageVisualizer rangeImageWidget;  
        
        float angRez = pcl::deg2rad(0.09f);
        float noiseLevel = 0.0f;
        float min_range = 0.0f;
        int border_size = 0;
        pcl::RangeImage::CoordinateFrame coordinateFrame = pcl::RangeImage::CAMERA_FRAME;
        Eigen::Affine3f sceneSensorPose ( Eigen::Affine3f::Identity());
        
//        float minVal = 0.0f;
//        float maxVal = 0.0f;
        
        
        int i_image(0);
        std::string ppm_base("output/depth_");        
        std::string ppm_filename;
        
//        std::stringstream ss;
        std::string ind_string;
        char i_image_char[10];
        
        while (!viewer.wasStopped ())
        {
            if (cloud_)
            {
                lock = true;
                FPS_CALC ("drawing");
                //the call to get() sets the cloud_ to null;
//                viewer.showCloud (getLatestCloud ());
                
                if (!cloud_->empty()){
                    try {
                        CloudConstPtr temp_cloud2;
                        temp_cloud2.swap (cloud_);

                        rangeImage.createFromPointCloud(*temp_cloud2, angRez, pcl::deg2rad(360.0f), pcl::deg2rad(180.0f), sceneSensorPose, coordinateFrame, noiseLevel, min_range, border_size);
                        
//                        rangeImageWidget.spin();
//                        std::cout << "Data: " <<  rangeImage.getRangesArray() << std::endl;   
                        
                        if (minRange_ < 0.0f)
                        {
                            rangeImage.getMinMaxRanges(minRange_, maxRange_);
                            rangeImage.setUnseenToMaxRange();                        
                            std::cout << "Min: " << minRange_ << " Max: " << maxRange_ << std::endl;
                        }
                        
//                        minRange_ = 3.25f;
//                        maxRange_ = 5.0f;

                        rangeImageWidget.visualize_selected_point = true;
                        rangeImageWidget.setRangeImage(rangeImage,  minRange_,  maxRange_, true);
//                        rangeImageWidget.setRangeImage(rangeImage,  minVal,  maxVal, true);                        
//                        rangeImageWidget.setRangeImage(rangeImage,  -std::numeric_limits<float>::infinity(),  std::numeric_limits<float>::infinity(), true);                        
                        

                        sprintf(i_image_char, "%04d", i_image);
                        ppm_filename = ppm_base + i_image_char + ".ppm";                                  
                        
                        std::cout << ppm_filename << std::endl;
                        rangeImageWidget.savePPM(ppm_filename,"");
//                        ss.str(std::string());
                        
                        i_image++; // increment image number
                    }catch (int e) {
                        std::cout << "Error" << std::endl;
                    }
                }
                lock = false;
                
            }
        }
        
        grabber_.stop();
    }
    
    pcl::visualization::CloudViewer viewer;
    pcl::ONIGrabber& grabber_;
    boost::mutex mtx_;
    CloudConstPtr cloud_;
};

void
usage(char ** argv)
{
    cout << "usage: " << argv[0] << " <path-to-oni-file> [framerate]\n";
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
        arg = argv[1];
        
        if (arg == "--help" || arg == "-h")
        {
            usage(argv);
            return 1;
        }
        
        if (argc >= 3)
        {
            frame_rate = atoi(argv[2]);
        }
        
        if (argc >= 5)
        {
            minRange_in = atoi(argv[3]);
            maxRange_in = atoi(argv[4]);            
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
        trigger.setInterval (1.0 / (double) frame_rate);
        trigger.registerCallback (boost::bind(&pcl::ONIGrabber::start, grabber));
        trigger.start();
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
