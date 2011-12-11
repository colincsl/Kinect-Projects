#include <pcl/cuda/cutil_math.h>
#include <pcl/cuda/common/eigen.h>
#include <iostream>
#include <cuda.h>
#include <vector>
#include <cutil.h>
#include <cuda_runtime_api.h>

#include <pcl/cuda/time_cpu.h>
#include <pcl/cuda/io/host_device.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_reference.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
//#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

#include "MLS.h"

//#include <math.h>
using namespace std;
using namespace thrust;
using namespace pcl::cuda;

using pcl::cuda::PointCloudAOS;
using pcl::cuda::Device;
using pcl::cuda::PointXYZRGB;


//struct neighborPtrs
//{
//	device_ptr<PointXYZRGB> nn[9];
//	int ind;
//	neighborPtrs():ind(0){};

//	void addPoint(device_ptr<PointXYZRGB> pt)
//	{
//		nn[ind] = pt;
//		ind++;
//	}
//};



void
thrustPCL_AOS(boost::shared_ptr<PointCloudAOS<Device> > cloud,
              PointCloudAOS<Device>::Ptr out,
              int nn_connectivity,
              float smoothness)
{

//            cudaSetDevice(1);

    const int size = 640*480;
    device_vector<int> indices(size);
    thrust::sequence(indices.begin(), indices.end());



//	// Init cuda memory for nn indices
//        int *nn_host, *nn_dev;
//	size_t ind_size = size*sizeof(int)*elements;
//	nn_host = (int*) malloc(ind_size);
//	cudaMalloc((void**) &nn_dev, ind_size);

//	int stride[] =  {-641, -640, -639, -1, 0, 1, 639, 640, 641};


//    int k=0;
//    for (int i=-nn_connectivity; i<= nn_connectivity; i++)
//    {
//        for (int j=-nn_connectivity; j<= nn_connectivity; j++)
//        {
//            stride[k] = i*640+j;
//            k++;
//        }
//    }
//	cudaMemcpy(nn_dev, nn_host, ind_size, cudaMemcpyHostToDevice);


//	cout << static_cast<PointXYZRGB>(*(cloud->points.begin())).x << endl;

        MovingLeastSquares kernel = MovingLeastSquares(cloud, nn_connectivity, smoothness);

        thrust::transform(make_zip_iterator(make_tuple(cloud->points.begin(), indices.begin())),
                          make_zip_iterator(make_tuple(cloud->points.end(), indices.end())), 
                          out->points.begin(),
                          kernel);

//        thrust::transform(make_zip_iterator(make_tuple(cloud->points.begin(), indices.begin())),
//                          make_zip_iterator(make_tuple(cloud->points.end(), indices.end())),
//                          out->points.begin(),
//                          kernel);


//        cudaFree(nn_dev);
//        free(nn_host);
	//cudaFree(indices);
}



/* ----------- Kernels ---------------*/

//        inline __host__ __device__
//        float dot(float3 x, float3 y)
//        {
//            return x.x*y.x + x.y*y.y + x.z*y.z;
//        }

//	__host__ __device__
//	float tTestKernel::operator()(const float& x, const float& y) const
//	{
//		return 0;
//	};

	//void PCLKernel::operator()(PointCloudAOS<Device>::Ptr x) const
//	PointXYZRGB PCLKernel::operator()(PointXYZRGB x) const
//	template <typename Tuple>
//	__host__ __device__
//	PointXYZRGB PCLKernel::operator()(const Tuple &data) const
//	{
//		//x.y = 0.0;
//		PointXYZRGB p = thrust::get<0>(data);
//		//p.x = 0.0;
//		neighborPtrs p2 = thrust::get<1>(data);
//
//		float x=0.0, y=0.0, z=0.0;
//		for (int i=0; i<9; i++)
//		{
//			PointXYZRGB nn = thrust::get<0>(p2.nn[0]);
//			x += nn.x;
//		}
//		p.x = x;
//
//		return p;
//	};


//        MovingLeastSquares(const CloudVar &data_, int nn_connect_, float smoothness_)
//        {
//            stride = new
//        };

    template <typename Tuple>
    __host__ __device__
    PointXYZRGB MovingLeastSquares::operator()(const Tuple &data) const
    {

        PointXYZRGB point = thrust::get<0>(data);
        int index = thrust::get<1>(data);

        PointXYZRGB point2, pOut, nn_;
        //point2 = points[index];


//        const int nn_count = 121;
//        int stride[nn_count];
\
//        int k=0;
//        for (int i=-nn_connectivity; i<= nn_connectivity; i++)
//        {
//            for (int j=-nn_connectivity; j<= nn_connectivity; j++)
//            {
//                stride[k] = i*640+j;
//                k++;
//            }
//        }

//        if (nn_connectivity == 1)
//        {
//            nn_count = 9;
//            stride =  {-641, -640, -639,\
//                             -1, 0, 1,\
//                             639, 640, 641};

////        const int nn_count = 25;
////        int stride[] =  {-1282, -1281, -1280, -1279, -1278,\
////                        -642, -641, -640, -639,-638, \
////                         -2, -1, 0, 1, 2, \
////                         638, 639, 640, 641, 642,\
////                         1278, 1279, 1280, 1281, 1282};

        const int nn_count = 49;
        int stride[] =  {-1923, -1922, -1921, -1920, -1919, -1918, -1917,\
                        -1283, -1282, -1281, -1280, -1279, -1278, 1977,\
                        -641, -642, -641, -640, -639,-638,-637, \
                         -3, -2, -1, 0, 1, 2, 3,\
                         638, 639, 640, 641, 642, 643,\
                         1277, 1278, 1279, 1280, 1281, 1282, 1283, \
                         1917, 1918, 1919, 1920, 1921, 1922, 1923};



        float3 centroid = make_float3(0.0,0.0,0.0);

        
        //vector<float3> neighbors(9); // don't use vector of cuda ??
        float3 neighbors[nn_count];

        // Find centroid
        int current_ind=0;
        int ind_count=0;
        int max_ind = 640*480;
        float thresh = .01*nn_connectivity;
            for (int i=0; i<nn_count; i++)
            {
                current_ind = index+stride[i];
                if (current_ind >= 0 && current_ind < max_ind)
                {
                    nn_ = points[current_ind];


//                    if (!isnan(nn_.x))// && (abs(nn_.x-point.x) < thresh) && (abs(nn_.y-point.y) < thresh) && (abs(nn_.z-point.z) < thresh))/*nn_.x==nn_.x)*/
                    if (nn_.x > 0.0);
                    {
                        centroid.x += nn_.x;
                        centroid.y += nn_.y;
                        centroid.z += nn_.z;

                        neighbors[ind_count] = make_float3(nn_.x, nn_.y, nn_.z);
//                        tmp_color = 255;
                        ind_count++;
                    }
                }
            }


//        if (ind_count < 1 || centroid.x != centroid.x)
        if (ind_count < 3)
        {
//            pOut.x = centroid.x;
//            pOut.y = centroid.y;
//            pOut.z = centroid.z;
//            pOut.rgb.r = 255;

            return pOut;
        }

        centroid.x /= (float)ind_count;
        centroid.y /= (float)ind_count;
        centroid.z /= (float)ind_count;            
        

        // Calculate covariance
        CovarianceMatrix cov;
        cov.data[0] = make_float3(0,0,0);
        cov.data[1] = make_float3(0,0,0);
        cov.data[2] = make_float3(0,0,0);

        float3 tmp_nn;
        for (int k=0; k<ind_count; k++)
        {
            tmp_nn = neighbors[k];
//            if (!isnan(tmp_nn.x) && !isnan(tmp_nn.y) && !isnan(tmp_nn.z) &&
//                !isnan(centroid.x) && !isnan(centroid.y) && !isnan(centroid.z))
//            {
                cov.data[0].x += (tmp_nn.x-centroid.x)*(tmp_nn.x-centroid.x);
                cov.data[0].y += (tmp_nn.x-centroid.x)*(tmp_nn.y-centroid.y);
                cov.data[0].z += (tmp_nn.x-centroid.x)*(tmp_nn.z-centroid.z);
                cov.data[1].y += (tmp_nn.y-centroid.y)*(tmp_nn.y-centroid.y);
                cov.data[1].z += (tmp_nn.y-centroid.y)*(tmp_nn.z-centroid.z);
                cov.data[2].z += (tmp_nn.z-centroid.z)*(tmp_nn.z-centroid.z);
//            }
        }
        cov.data[0].x /= (ind_count-1);
        cov.data[0].y /= (ind_count-1);
        cov.data[0].z /= (ind_count-1);
        cov.data[1].y /= (ind_count-1);
        cov.data[1].z /= (ind_count-1);
        cov.data[2].z /= (ind_count-1);

        // fill in the lower triangle (symmetry)
        cov.data[1].x = cov.data[0].y;
        cov.data[2].x = cov.data[0].z;
        cov.data[2].y = cov.data[1].z;

        CovarianceMatrix evecs;
        float3 evals;

//        if (1)//(cov.data[0].x > 0 && cov.data[0].y > 0 && cov.data[0].z > 0)
//        {
            pcl::cuda::eigen33 (cov, evecs, evals);
//        } else {
//            pOut.x = centroid.x;
//            pOut.y = centroid.y;
//            pOut.z = centroid.z;
//            pOut.rgb.b = (unsigned int)250;
//            return pOut;
//        }


    // model_coeffs.head = eigenvector (in serial mls)

        float3 pointC;
        float3 normal = evecs.data[0];
//        float3 model_vec = evecs.data[2];
        float eigenvalue = evals.z;

        pointC.x = point.x; pointC.y = point.y; pointC.z = point.z;
        float model_coeff = -1*(normal.x*centroid.x + normal.y*centroid.y + normal.z*centroid.z);

        float distance = pointC.x*normal.x + pointC.y*normal.y + pointC.z*normal.z + model_coeff;
        pointC -= distance * normal;

        float curvature = cov.data[0].x+cov.data[1].y+cov.data[2].z; // curv = tr(covariance)
        if (curvature != 0) curvature = fabs(eigenvalue / curvature);

        float nn_dist[nn_count];
        for (int i=0; i<ind_count; i++)
        {
            neighbors[i] -= pointC;
            nn_dist[i] = neighbors[i].x*neighbors[i].x + neighbors[i].y*neighbors[i].y + neighbors[i].z*neighbors[i].z;
        }

        // Init polynomial params - assume 3 coeffs for now
        const int nr_coeff = 3; // number of coeffs in polynomial
        const float sqr_gauss_param = smoothness;

        float weight_vec[nn_count];
        float f_vec[nn_count];
        float3 c_vec;
        float3 P[nn_count];
        float3 P_weight[nn_count];
        //float3 P_weight_Pt[3];

        //Local coordinate system
        float3 v = unitOrthogonal(normal);
        float3 u = cross(normal, v);

        float u_coord, v_coord, u_pow, v_pow;
        for (int i=0; i<ind_count; i++)
        {
            // Compute weight
            weight_vec[i] = exp(-nn_dist[i] / sqr_gauss_param);
            // Transform coords
            u_coord = neighbors[i].x*u.x + neighbors[i].y*u.y +  neighbors[i].z*u.z;
            v_coord = neighbors[i].x*v.x + neighbors[i].y*v.y +  neighbors[i].z*v.z;
            f_vec[i]= dot(neighbors[i], normal);

            u_pow = 1;
            for(int i2=0; i2<nr_coeff-1; i2++)
            {
                v_pow=1;
                P[i].x = u_pow*v_pow;
                v_pow *= v_coord;
                P[i].y = u_pow*v_pow;
                v_pow *= v_coord;
                P[i].z = u_pow*v_pow;
                v_pow *= v_coord;

                u_pow *= u_coord;
            }
        }

//        // Computing coefficients
//        P_weight = P * weight_vec.asDiagonal ();
//        P_weight_Pt = P_weight * P.transpose ();
//        c_vec = P_weight * f_vec;
//        P_weight_Pt.llt ().solveInPlace (c_vec);

        // P is NNx3
        // P_weight is NNx3
        // P_weight_Pt is 3x3
        for (int i=0; i<ind_count; i++)
        {
            P_weight[i].x = P[i].x*weight_vec[i];
            P_weight[i].x = P[i].x*weight_vec[i];
            P_weight[i].x = P[i].x*weight_vec[i];
        }

        c_vec.x=0.0; c_vec.y=0.0; c_vec.z=0.0;
        for (int i=0; i<ind_count; i++)
        {
            c_vec.x += P_weight[i].x*f_vec[i];
            c_vec.y += P_weight[i].y*f_vec[i];
            c_vec.z += P_weight[i].z*f_vec[i];
        }


        pointC += c_vec.x*normal;

        pOut.x = pointC.x;
        pOut.y = pointC.y;
        pOut.z = pointC.z;
        pOut.rgb.b = 150;
        pOut.rgb.r = 150;

        return pOut;
    };
