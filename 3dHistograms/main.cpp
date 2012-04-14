


#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/histogram_visualizer.h>

#include <boost/ptr_container/ptr_vector.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace pcl::io;
using namespace cv;
using pcl::PointXYZRGB;
//using pcl::Histogram;

#define VIZ 1
#define HOME 0

template <typename PointType>
class Histograms
{
  public:
    Histograms () : viewer ("KinectGrabber"), /*histViz_x(),*/ histWindow(Mat::zeros(300, 800, CV_8UC1))
    {
//        hist_x_h.points.resize(1);
//        for (int i=0; i<N; i++) hist_x_h.points[0].histogram[i] = 0;
//        histViz_x.addFeatureHistogram(hist_x_h, N, "X");
//        histViz_x.setBackgroundColor(0, 0, 0);

        namedWindow("3D Histogram");
        histMax = 300;
    };


    typedef pcl::PointCloud<PointType> Cloud;
    typedef typename Cloud::Ptr CloudPtr;
    typedef typename Cloud::ConstPtr CloudConstPtr;
    
    void cloud_cb_ (const CloudConstPtr& cloud)
    {

//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr output (new pcl::PointCloud<pcl::PointXYZRGB>);
        int pxCount = 640*480;
        const int bins = 100;
        int px_bin_x, px_bin_y, px_bin_z;
        
        float min_x, min_y, min_z;
        min_x = min_y = min_z = 1000.0; //min kinect dist
        float max_x, max_y, max_z;
        max_x = max_y = max_z = -1000.0; //max kinect dist
        
        // Find min/max
        for (size_t i = 0; i < cloud->points.size (); ++i)
        {
            if (min_x > cloud->points[i].x)
                min_x = cloud->points[i].x;
            else if (max_x < cloud->points[i].x)
                max_x = cloud->points[i].x; 
            
            if (min_y > cloud->points[i].y)
                min_y = cloud->points[i].y;
            else if (max_y < cloud->points[i].y)
                max_y = cloud->points[i].y;
            
            if (min_z > cloud->points[i].z)
                min_z = cloud->points[i].z;
            else if (max_z < cloud->points[i].z)
                max_z = cloud->points[i].z;            
        }
        float max_ = max(max(max_x, max_y), max_z);
        float min_ = min(min(min_x, min_y), min_z);
        float inc = (max_-min_)/bins;
        
        vector<int> hist_x(bins, 0);
        vector<int> hist_y(bins, 0);
        vector<int> hist_z(bins, 0);        
        
        for (size_t i = 0; i < cloud->points.size (); ++i)
        {
            if (!isnan(cloud->points[i].y))
            {
                px_bin_x =  round(((cloud->points[i].x - min_x)/inc));
                px_bin_y =  round(((cloud->points[i].y - min_y)/inc));
                px_bin_z =  round(((cloud->points[i].z - min_z)/inc));                

                if (px_bin_x < bins && px_bin_x >= 0)
                    hist_x[px_bin_x]++;
                if (px_bin_y < bins && px_bin_y >= 0)
                    hist_y[px_bin_y]++;                
                if (px_bin_z < bins && px_bin_z >= 0)
                    hist_z[px_bin_z]++;
            }
        
        }
        
        
        cout << "----- Histogram -----" << endl;
        for (int i=0; i<bins; i++)
            cout << hist_x[i] << "\t\t" << hist_y[i] << "\t\t" << hist_z[i] << endl;
        cout << "---------------------" << endl;
        
        // Find max value(s)
        int max_hist_z = 0;
        for (int i=0; i<bins; i++)
        {
            if (hist_z[i] > max_hist_z)
                max_hist_z = hist_z[i];
        }
        
        // Display in histWindow frame
        float z_conversion = 250.0 / (float)max_hist_z;
        float width_conversion = 800 / (bins+1);
        histWindow = Mat::zeros(300,800, CV_8UC1);
        for (int i_ind = 0; i_ind < bins; i_ind++)
        {
            int tmp_height = hist_z[i_ind]*z_conversion;
            for (int i=0; i < tmp_height; i++)
            {
                histWindow.at<char>(histMax-i, i_ind*width_conversion+width_conversion/2) = (char)255;
            }
        }
        
        // the window freezes up!
        imshow("3D Histogram", histWindow);
        waitKey(33);
        
        
        viewer.showCloud (cloud);
        
        return;
    };
    
    void run (const std::string& device_id)
    {

      pcl::Grabber* interface = new pcl::OpenNIGrabber(device_id);

      boost::function<void (const CloudConstPtr&)> f = boost::bind (&Histograms::cloud_cb_, this, _1);
      boost::signals2::connection c = interface->registerCallback (f);

      pcl_sleep(.5);
      interface->start ();


      while (true)
      {
        pcl_sleep (.1);
      }

      interface->stop ();
    };

    pcl::visualization::CloudViewer viewer;
    boost::mutex mutex_;    
    Mat histWindow;
    int histMax;
    //    pcl::visualization::PCLHistogramVisualizer histViz_x;
    //    pcl::PointCloud<pcl::Histogram<N> > hist_x_h;    

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

    int bins = 10;
    Histograms<PointXYZRGB> v;
    v.run(device_id);

    return 0;
}
