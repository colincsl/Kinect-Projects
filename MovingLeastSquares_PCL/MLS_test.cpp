


#include <cuda.h>

#include "pcl/cuda/io/cloud_to_pcl.h"
#include "pcl/cuda/io/disparity_to_cloud.h"

#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/cuda/time_cpu.h>
#include <pcl/cuda/io/host_device.h>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/cloud_viewer.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include <boost/ptr_container/ptr_vector.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace pcl::cuda;
using namespace pcl::io;
using pcl::cuda::PointCloudAOS;
using pcl::cuda::Device;
using namespace boost;



#include "MLS.h"

#define MEGA 1048576
#define VIZ 1
#define BALAUR 1

template <typename PointType>
class gpuIterate
{
  public:
	#if VIZ
         gpuIterate (int nn_connect_, float smoothness_) : viewer ("KinectGrabber"),
                                                           saved(false),
                                                           nn_connectivity(nn_connect_),
                                                           smoothness(smoothness_){}
	#else
     gpuIterate (int nn_connect_, float smoothness_) :     saved(false),
                                                           nn_connectivity(nn_connect_),
                                                           smoothness(smoothness_){}
	#endif


     typedef pcl::PointCloud<PointType> Cloud;
     typedef typename Cloud::Ptr CloudPtr;
     typedef typename Cloud::ConstPtr CloudConstPtr;
    template <template <typename> class Storage>
    void cloud_cb_ (const CloudConstPtr& cloud)
    {
        { //time
        pcl::cuda::ScopeTimeCPU t ("GPU fcn time:");

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr output (new pcl::PointCloud<pcl::PointXYZRGB>);
        PointCloudAOS<Host> data_host;
        PointCloudAOS<Device>::Ptr data;
        PointCloudAOS<Device>::Ptr data_out;
//        int tmp_stride[] =  {-641, -640, -639, -1, 0, 1, 639, 640, 641};
//        vector<int> stride(tmp_stride, tmp_stride+9);
        int pxCount = 640*480;

        data_host.points.resize (cloud->points.size());
        for (size_t i = 0; i < cloud->points.size (); ++i)
        {
          PointXYZRGB pt;
          pt.x = cloud->points[i].x;
          pt.y = cloud->points[i].y;
          pt.z = cloud->points[i].z;
          // Pack RGB into a float
          pt.rgb = *(float*)(&cloud->points[i].rgb);
          data_host.points[i] = pt;
        }
        data_host.width = cloud->width;
        data_host.height = cloud->height;
        data_host.is_dense = cloud->is_dense;

    	//boost::mutex::scoped_lock l(mutex_);


		size_t free, total;
		cudaMemGetInfo(&free, &total);
		if (free/MEGA < 15)
		{
			cout << "Not enough memory!   ";
			cout << "Free: " << free/MEGA << " of " << total/MEGA << " mb" << endl;
			return;
		}
		//data_host.points.resize (cloud->points.size());
		//data_out->points.resize(cloud->points.size());


//         { //time
//            pcl::cuda::ScopeTimeCPU t ("GPU fcn time:");
            data = toStorage<Host, Storage> (data_host); 	//input
            data_out = toStorage<Host, Storage> (data_host);//output
            thrustPCL_AOS(data, data_out, nn_connectivity, smoothness);

//         }
//
// 		for (int i=0; i<size; i++) cout << Out[i] << endl;
 		//thrust::transform(data.begin(), data.end(), Y.begin(), tTestKernel(a));
//    		cudaMemGetInfo(&free, &total);
//    		cout << "Free: " << free/MEGA << endl;
//    		cout << "Total: " << total/MEGA << endl;

        pcl::cuda::toPCL (*data_out, *output);





      if (!saved)
      {
    	  std::string filename_base = "test";
    	  std::string filename_orig = filename_base;
    	  std::string filename_proc = filename_base;
    	  filename_orig.append(".pcd");
    	  filename_proc.append("_out.pcd");

    	  savePCDFile(filename_orig, *cloud);
    	  savePCDFile(filename_proc, *output);
    	  saved = true;
    	  cout << "PCD file saved to: " << filename_proc << endl;
      }
      } // end time
	#if VIZ
//         viewer.showCloud (output);
	#endif

      return;
    }
    
    void run (const std::string& device_id)
    {

      pcl::Grabber* interface = new pcl::OpenNIGrabber(device_id);

      boost::function<void (const CloudConstPtr&)> f = boost::bind (&gpuIterate::cloud_cb_<Device>, this, _1);
//      boost::function<void (const Cloud&)> f = boost::bind (&gpuIterate::cloud_cb_, this, _1);

      boost::signals2::connection c = interface->registerCallback (f);

      pcl_sleep(1);
      interface->start ();


      while (true)
      {
        pcl_sleep (1);
      }

      interface->stop ();
    }

    pcl::cuda::DisparityToCloud d2c;
    bool saved;
    int nn_connectivity;
    float smoothness;
    #if VIZ
    pcl::visualization::CloudViewer viewer;
    #endif
    boost::mutex mutex_;

};

int main (int argc, char** argv)
{
	#if BALAUR
            std::string device_id = "B00364201091048B";
	#else
            std::string device_id = "B00361225874048B";
	#endif

        int nn_connectivity=1;
        float smoothness=.01;

		if (argc >= 2)
		{
                        //device_id = argv[1];
		}
		if (argc >= 3)
		{
                        nn_connectivity = atoi (argv[2]);
		}
                if (argc >= 3)
                {
                        smoothness = atoi (argv[3]);
                }

        gpuIterate<pcl::PointXYZRGB> v (nn_connectivity, smoothness);
        v.run (device_id);

  return 0;
}
