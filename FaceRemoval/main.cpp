

//General dependencies

#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <time.h>
using namespace std;


//PCL dependencies

#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <boost/ptr_container/ptr_vector.hpp>
#include <pcl/io/io.h>
using namespace pcl::io;
using namespace boost;
//using pcl::PointXYZRGB;
using namespace pcl;

// OpenCV depenedencies

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

// Globals

#define VIZ 1
#define HOME 0
//String face_cascade_name = "/Users/colin/libs/kinect/opencv/opencv/data/lbpcascades/lbpcascade_frontalface.xml";
String face_cascade_name = "/Users/colin/libs/kinect/opencv/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
String profile_cascade_name = "/Users/colin/libs/kinect/opencv/opencv/data/haarcascades/haarcascade_profileface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;



template <typename PointType, int N>
class FaceRemoval
{
  public:
    #ifdef VIZ
        FaceRemoval ():img(480, 640, CV_8UC1), viewer ("Cloud")
        {
            imshow( "win", img );
//            cloud.points.resize(480*640);
            cloud.reset(new Cloud);
            cloud->points.resize(480*640);
        };
    #else
        FaceRemoval ():img(480, 640, CV_8UC1) /* viewer ("KinectGrabber")*/
        {
            imshow( "win", img );
        };
    #endif
    


    typedef pcl::PointCloud<PointType> Cloud;
    typedef typename Cloud::Ptr CloudPtr;
    typedef typename Cloud::ConstPtr CloudConstPtr;
    
    void cloud_cb_ (const CloudConstPtr& cloud_)    
    {


//        boost::mutex::scoped_lock lock (mutex_);
        if (!mutex_.try_lock())
            return;

//        cloud.reset(cloud_);
        copyPointCloud(*cloud_, *cloud); 
        int height = 480;
        int width = 640;
        int index=0;
        unsigned int color;

        for (int j=0; j<height; j++)
        {
            for (int i=0; i<width; i++)
            {
                color = ((unsigned int)cloud->points[index].r+(unsigned int)cloud->points[index].g+(unsigned int)cloud->points[index].b)/3;
                img.at<char>(j,i) = (char)color;                
//                img.at<char>(j,i) = cloud->points[index].r;
                
                index++;
            }
        }
        
        std::vector<Rect> faces;
        faces = detectFaces();
        
        if (faces.size() >0)
        {
            for (int i=0; i<faces.size(); i++)
            {
                
                for (int y = faces[i].y; y<faces[i].y+faces[i].height; y++)
                {                
                    for (int x = faces[i].x; x<faces[i].x+faces[i].width; x++)
                    {
                        cloud->points[y*width+x].r = (char)0;
                        cloud->points[y*width+x].b = 0;
                        cloud->points[y*width+x].g = 255;                        
                    }
                }
                
            }
        }
        
        
        cout << "A" << endl;
        
        #ifdef VIZ
        viewer.showCloud (cloud);
//        viewer.addPointCloud (cloud);        
        
//        viewer.saveScreenshot (filename);
        #endif

        string filename = "cloud_";        
        time_t rawtime;
        time(&rawtime);
        filename.append(ctime(&rawtime));
        filename.append(".pcd");


        pcdWrite.writeBinaryCompressed(filename, *cloud);            
//        pcl::io::savePCDFileASCII ("ascii.pcd", cloud);
//        pcl::io::savePCDFileBinary ("bin.pcd", cloud);
        
        mutex_.unlock();
        
        return;
    }
    
    /* ------------------- FACES --------------------- */
    
    inline std::vector<Rect> detectFaces()
    {
        std::vector<Rect> faces;
        std::vector<Rect> faces_profile;        
        
        equalizeHist(img, img);
        
        // Params: image, faceVector, relative scale, req. neighbor #, flag, minSize
        face_cascade.detectMultiScale(img, faces, 1.1, 1, 0, Size(30, 30));
        face_cascade.detectMultiScale(img, faces_profile, 1.1, 1, 0, Size(30, 30)); 
        
        while (faces_profile.size() > 0)
        {
            faces.push_back(faces_profile.back());
            faces_profile.pop_back();
        }
        
        for (int i=0; i<faces.size(); i++)
        {
//            Mat faceROI = img(faces[i]);
            Point center(faces[i].x + faces[i].width*.5, faces[i].y+faces[i].height*.5);
            ellipse(img, center, Size(faces[i].width*.5, faces[i].height*.5), 0, 0, 360, Scalar(250, 0, 0), 2, 8, 0);
            cout << "Face at " << faces[i].x << " " << faces[i].y << endl;
        }
        
        return faces;
    }
    
    bool run (const std::string& device_id)
    {
        
        if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading Faces\n"); return false; };
        if( !profile_cascade.load( profile_cascade_name ) ){ printf("--(!)Error loading Profiles\n"); return false; };        
//        if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading Eyes\n"); return false; };

        
        pcl::Grabber* interface = new pcl::OpenNIGrabber(device_id);

        boost::function<void ( const CloudConstPtr&)> f = boost::bind (&FaceRemoval::cloud_cb_, this, _1);
        boost::signals2::connection c = interface->registerCallback (f);

        interface->start ();


        while (true)
        {
            pcl_sleep (1);
            imshow( "win", img );
            waitKey(33);
            cout << "img" << endl;
        }

        interface->stop ();

        return true;
    };

    // Class vars
    #ifdef VIZ
        pcl::visualization::CloudViewer viewer;
//    pcl::visualization::PCLVisualizer viewer;
    
    #endif
    boost::mutex mutex_;
//    Cloud cloud;
//    CloudConstPtr cloud_ptr;
    CloudPtr cloud;    
    Mat img;
    PCDWriter pcdWrite;
    
    CascadeClassifier face_cascade;
    CascadeClassifier profile_cascade;    
//    CascadeClassifier eyes_cascade;

};

int main (int argc, char** argv)
{
    #if HOME
        std::string device_id = "B00361225874048B";
    #else (CIRL #1)
        std::string device_id = "A00367905072044A";
    #endif

    if (argc >= 2)
    {
        device_id = argv[1];
    }
    if (argc >= 3)
    {
//        nn_connectivity = atoi (argv[2]);
    }
    int bins = 10;
    namedWindow ("win", 1 );
    
    FaceRemoval<PointXYZRGB, 10> v;
    v.run(device_id);

    return 0;
}
