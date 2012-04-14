/*	
 * Author: Colin Lea (colincsl@gmail.com)
 * December 2011
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

#define MEGA 1

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
    int i_image;
    Mat cvImg;
    
    
    SimpleONIViewer(pcl::ONIGrabber& grabber, int nearestNeighbor_, float minRange=-1.0f, float maxRange=-1.0f, std::string fileout_=""): grabber_(grabber), nearestNeighbor(nearestNeighbor_), lock(false), minRange_(minRange), maxRange_(maxRange), cvImg(Size(640, 480), CV_8U), fileout(fileout_), i_image(0) 
    {
//        namedWindow("Win");
    }
    
    /**
     * @brief Callback method for the grabber interface
     * @param cloud The new point cloud from Grabber
     */
    void
    cloud_cb_ (const CloudConstPtr& cloud)
    {

        FPS_CALC ("callback");
        
        // Lock pointcloud so another thread can't interrupt
//        mtx_.lock();
        boost::mutex::scoped_lock lock(mtx_);
//        cout << "locking" << endl;
        
        cloud_ = cloud;
        float tmp_z;
        
        // If not set with input args, get min/max range
        if (minRange_ < 0.0f)
        {            
            for (int i=0; i<480*640; i++)        
            {
                tmp_z = cloud_->points[i].z;
                if (tmp_z > maxRange_) maxRange_ = tmp_z; 
                if (tmp_z < minRange_) minRange_ = tmp_z;                                    
            }            
            std::cout << "Min: " << minRange_ << " Max: " << maxRange_ << std::endl;            
            
        }    

        // Find path to output image
        std::string img_base("");
        
        if (fileout == "")
            img_base.insert(0,"output/depth_");
        else
            img_base.insert(0,fileout);
        
        std::string img_filename;        
        std::string ind_string;
        char i_image_char[20];
        sprintf(i_image_char, "/depth_%04d", i_image);
        img_filename = img_base + i_image_char + ".jpg";                                  
        
//        std::cout << img_filename << std::endl;

        // Set depth info from rangeimage to OpenCV image
        char rgb_val;
        float maxDist = minRange_+maxRange_;
        
        for (int y=0; y<480; y++)        
        {
            for (int x=0; x<640; x++)
            {
                tmp_z = cloud_->points[y*640+x].z;
                
                if (tmp_z > maxRange_) tmp_z = maxRange_;
                
                rgb_val = saturate_cast<char>(256.0 - (tmp_z-minRange_)/(maxRange_-minRange_)*256.0);
                cvImg.at<char>(y, x) = rgb_val;
            }
        }
        
        
        if (nearestNeighbor)
        {
            Mat distX, distY;
            distX = Mat::zeros(Size(640, 480), CV_16S);
            distY = Mat::zeros(Size(640, 480), CV_16S);
            int dx, dy;
            
            for (int y=1; y<480; y++)        
            {
                for (int x=1; x<640; x++)
                {
                    tmp_z = cloud_->points[y*640+x].z;
//                    tmp_z = cvImg.at<char>(y,x);
                    if (tmp_z != tmp_z)
                    {
                        if (cvImg.at<char>(y,x-1))                        
                            distX.at<int>(y,x) = -1;
                        else if (distX.at<int>(y,x-1))
                            distX.at<int>(y,x) = distX.at<int>(y,x-1)-1;
                        else /* if left dist is 0 */
                            distX.at<int>(y,x) = 0;
                        
                        if (cvImg.at<char>(y-1,x))
                            distY.at<int>(y,x) = -1;
                        else if (distY.at<int>(y-1,x))
                            distY.at<int>(y,x) = distY.at<int>(y-1,x)-1;
                        else
                            distY.at<int>(y,x) = 0;
                    }
                }
            }
            
            for (int y=480-1; y>=0; y--)        
            {
                for (int x=640-1; x>=0; x--)
                {
                    tmp_z = cloud_->points[y*640+x].z;

                    if (tmp_z != tmp_z)
                    {
                        if (cvImg.at<char>(y,x+1))
                            distX.at<int>(y,x) = 1;
                        else
                        {
                            if (distX.at<int>(y,x+1))
                            {   
                                if (abs(distX.at<int>(y,x)) > distX.at<int>(y,x+1)+1)
                                    distX.at<int>(y,x) = distX.at<int>(y,x+1)+1;
                            }
                        }
                        
                        if (cvImg.at<char>(y+1,x))
                            distY.at<int>(y,x) = 1;
                        else
                        {
                            if (distY.at<int>(y+1,x))
                            {
                                if (abs(distY.at<int>(y,x)) > distY.at<int>(y+1,x)+1)
                                    distY.at<int>(y,x) = distY.at<int>(y+1,x)+1;                            
                            }
                        }
                    
                        // Apply to the original image
                        // But first find if vertical or horizontal pixel is closer
                        if (abs(distX.at<int>(y,x)) < abs(distY.at<int>(y,x)))
                        {
                            dx = distX.at<int>(y,x);
                            dy = 0;
                        }
                        else
                        {
                            dx = 0;                            
                            dy = distY.at<int>(y,x);
                        }
                        cvImg.at<char>(y,x) = cvImg.at<char>(y+dy,x+dx);                    
                    
                    
                    }
                }
            }            
            
       
        }
        
        
        
        imwrite(img_filename, cvImg);        
        i_image++; // increment image number

        time = pcl::getTime();        
        tmpLock = false;
//        cout << "Threadi: " << boost::this_thread::get_id() << endl;
//        std::cout << img_filename << std::endl;
//        mtx_.unlock();
    }
    
    
    /**
     * @brief starts the main loop
     */
    void
    run()
    {
        boost::function<void (const CloudConstPtr&) > f = boost::bind (&SimpleONIViewer::cloud_cb_, this, _1);        
        boost::signals2::connection c = grabber_.registerCallback (f);
        
        tmpLock = false;
        grabber_.start();
        time = pcl::getTime();
        while (pcl::getTime()-time < 3.0) 
        {    
//            cout << "Lock: " << tmpLock << " " << boost::this_thread::get_id() << endl;
            // Trigger the next frame
            
            // Add timed mutex. Maybe not all 'frames' have a cloud?
            
            if (!tmpLock || pcl::getTime()-time > 0.5)// && mtx_.try_lock())
            {
                tmpLock = true;
                grabber_.start();
            }
            boost::this_thread::sleep(boost::posix_time::seconds(.01));

        }
        
        cout << "Outputted " << i_image-1 << " frames." << endl;
        
    }
    
    pcl::ONIGrabber& grabber_;
    boost::mutex mtx_;
    boost::mutex mtx_timed;
    CloudConstPtr cloud_;
    std::string fileout;
    double time;
    int nearestNeighbor;
    bool tmpLock;
};

void
usage(char ** argv)
{
    cout << "usage: " << argv[0] << " <path-to-oni-file> [min range] [max range] [path-to-jpg-folder] \n";
    cout << argv[0] << " -h | --help : shows this help" << endl;
    return;
}

int
main(int argc, char ** argv)
{
    std::string arg("");
    std::string fileout("");
    
    float minRange_in = -1.0f;
    float maxRange_in = -1.0f;
    int nearestNeighbor(0);
    
    unsigned frame_rate = 0;
    if (argc < 2)
    {
        usage(argv);
        return 1;
    }
    
    arg = argv[1]; // Filename
    
    if (argc >= 4)
    {
        minRange_in = atof(argv[2]);
        maxRange_in = atof(argv[3]);           
        cout << "Min Range: " << minRange_in << "\t Max Range: " << maxRange_in;
        cout << " Resolution: " << (maxRange_in-minRange_in)/256.0*10 << " cm" << endl;
    }
    if (argc >= 5)
        fileout = argv[4];

    if (argc >= 6)
        nearestNeighbor = atoi(argv[5]);
    
    
    pcl::ONIGrabber* grabber;
    grabber = new  pcl::ONIGrabber(arg, false, false);
    
    SimpleONIViewer<pcl::PointXYZRGB> v(*grabber, nearestNeighbor, minRange_in, maxRange_in, fileout);
    v.run();
    
    return (0);
}
