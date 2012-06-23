/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: oniExKinectTest.cpp 3048 2011-10-11 13:40:46Z adeguet1 $

  Author(s):  Kelleher Guerin and Simon Leonard
  Created on: 2008

  (C) Copyright 2006-2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <sawOpenNI/osaOpenNI.h>
#include <cisstCommon/cmnLogger.h>
#include <cisstCommon/cmnGetChar.h>
#include <cisstCommon/cmnPath.h>
#include <cisstVector/vctDynamicMatrix.h>
#include <cisstOSAbstraction/osaSleep.h>

#include <iostream>
#include <fstream>
#include <sstream>

//#include <cisstCommon/cmnGetChar.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <pthread.h>

using namespace std;


// Keyboard handler
char keyInput = 'a';
char toggleInput = 0;
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;

void* getKey(void*)
{
    char tmp;
    while (true)
    {
        tmp = cmnGetChar();
        pthread_mutex_lock(&mutex1);
        toggleInput = 1;
        if (tmp != char(13))
        {
            keyInput = tmp;
//            cout << "(Key:" << keyInput << ")";
        }
        pthread_mutex_unlock(&mutex1);        
    }
};

struct keyBoardInput
{
    
    keyBoardInput() {
        pthread_t thread_id;
        int rc = 0;
        rc = pthread_create(&thread_id, NULL, getKey, NULL);
        if (rc) 
            cerr << "Error creating thread" << endl;
    }
};

int main(int argc, char** argv){
    
    // Setup joint logger
    int logCount(0);
    ofstream logger;    

	cmnLogger::SetMask(CMN_LOG_ALLOW_ALL);
	cmnLogger::SetMaskFunction(CMN_LOG_ALLOW_ALL);
	cmnLogger::SetMaskDefaultLog(CMN_LOG_ALLOW_ALL);

    int numusers = 6;
    if (argc == 2) {
        sscanf(argv[1], "%d", &numusers);
    }


    cmnPath path;
    path.Add(".");
    std::string configFile = path.Find("SamplesConfig.xml");
    if (configFile == "") {
        std::cerr << "can't find file \"SamplesConfig.xml\" in path: " << path << std::endl;
        exit (-1);
    }
	osaOpenNI kinect(numusers);
    kinect.Configure(configFile);
    if (0 < numusers) {
        kinect.InitSkeletons();
    }

	while (true) {
        // Wait and Update All
        kinect.Update(WAIT_AND_UPDATE_ALL);
        if (0 < numusers) {
            kinect.UpdateUserSkeletons();
        }

        
        { // Get RGB
            vctDynamicMatrix<unsigned char> rgb;
            if (kinect.GetRGBImage(rgb) != osaOpenNI::ESUCCESS) {
                CMN_LOG_RUN_ERROR << "Failed to get RGB image" << std::endl;
                return -1;
            }

            std::ofstream ofs("rgb");
            for (size_t r=0; r<rgb.rows(); r++) {
                for (size_t c=0; c<rgb.cols(); c++) {
                    ofs << (int)rgb[r][c] << " ";
                }
                ofs << std::endl;
            }
            ofs.close();
        }

        if (0)
        { // Get displacement map
            vctDynamicMatrix<double> depth;
            if (kinect.GetDepthImageRaw(depth) != osaOpenNI::ESUCCESS) {
                CMN_LOG_RUN_ERROR << "Failed to get RGB image" << std::endl;
                return -1;
            }
            std::ofstream ofs("depth");
            ofs << depth;
            ofs.close();
        }

        { // Get range image
            vctDynamicMatrix<double> range;
            std::vector< vctFixedSizeVector<unsigned short, 2> > pixels;
            if (kinect.GetRangeData(range, pixels) != osaOpenNI::ESUCCESS) {
                CMN_LOG_RUN_ERROR << "Failed to get RGB image" << std::endl;
                return -1;
            }
            std::ofstream ofs("range");
            ofs << range;
            ofs.close();
        }

        if (0 < numusers) {
            std::vector<osaOpenNISkeleton*> skeletons = kinect.UpdateAndGetUserSkeletons();
        }
        std::cerr << "*";
        
//        int keyInput;

//        cv::namedWindow("Empty");
//        cv::Mat fakeImg;
//        fakeImg.create(1, 1, CV_8U);
//        cv::imshow("Empty", fakeImg);
        
        keyBoardInput *keyboard = new keyBoardInput();
        
        while(0)
        {
            kinect.Update(WAIT_AND_UPDATE_ALL);      
            
            
//            if (toggleInput)
//                cout << "(Key:" << keyInput << ")" << endl;

            if (toggleInput) // 32 = space bar
            {    
                // Start logger (if closed)
                if ( !logger.is_open() ){
                    stringstream fileName;
                    fileName << "jointLog" << logCount << ".txt";
                    logger.open(fileName.str().c_str());
                    cout << " >>>>> Gesture" << logCount << " started ... ";
//                    delete &fileName;
                } else {
                // Stop
                    logger.close();
                    logCount++;
                    cout << " Gesture finished <<<<<<< " << endl;                    
                    
                }
                toggleInput = 0;
            }
            
            
            if (0 < numusers && logger.is_open()) {      
                std::vector<osaOpenNISkeleton*> skeletons = kinect.UpdateAndGetUserSkeletons();
                
                for (int i=0; i<5; i++)
                {  
                    if (skeletons[i]->exists) {
                        std::cout << "Skeleton: " << i << std::endl;
                        logger << "Skeleton " << i << endl;
                        for (int j=0; j<25; j++) {
//                            std::cout << skeletons[i]->points3D[j] << std::endl;
                            logger << skeletons[i]->points3D[j] << endl;
                        }
                        
                    }
                }

            }                
            
        }
	}

	return 0;
}
