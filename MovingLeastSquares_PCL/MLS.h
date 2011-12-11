
#ifndef ITERATE_H
#define ITERATE_H

//#include <iostream>
//#include <cuda.h>
//#include <cutil.h>
//#include <cuda_runtime_api.h>


using namespace std;

#include <pcl/cuda/io/host_device.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_reference.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>


using namespace pcl::cuda;
using namespace thrust;
using pcl::cuda::PointCloudAOS;
using pcl::cuda::Device;
using pcl::cuda::PointXYZRGB;

void thrustPCL();
void thrustPCL_AOS(PointCloudAOS<Device>::Ptr x, PointCloudAOS<Device>::Ptr y, int nn_connect, float smoothness);

struct tTestKernel
{
	const float a;

	tTestKernel(float _a):a(_a){};

	__host__ __device__
	float operator()(const float& x, const float& y) const;

};

struct PCLKernel
{

	PCLKernel(){};

//float operator()(PointCloudAOS<Device>::Ptr x) const;
	template <typename Tuple>
	__host__ __device__
	PointXYZRGB operator()(const Tuple &x) const;	
	

};

struct MovingLeastSquares
{
    typedef boost::shared_ptr<PointCloudAOS<Device> > CloudVar;

    const PointXYZRGB *points;
    int nn_connectivity;
    int nn_count_;
    float smoothness;

//    __shared__ int *stride;

    MovingLeastSquares(const CloudVar &data_, int nn_connect_, float smoothness_):
                            points(raw_pointer_cast<const PointXYZRGB> (&data_->points[0])),
                            nn_connectivity(nn_connect_),
                            smoothness(smoothness_),
                            nn_count_(nn_connect_*nn_connect_){}

//float operator()(PointCloudAOS<Device>::Ptr x) const;
	template <typename Tuple>
	__host__ __device__
	PointXYZRGB operator()(const Tuple &x) const;


};



#endif

