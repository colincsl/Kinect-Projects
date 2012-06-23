/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: oniExKinectView.cpp 3048 2011-10-11 13:40:46Z adeguet1 $

  Author(s):  Balazs Vagvolgyi
  Created on: 2011

  (C) Copyright 2006-2011 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstCommon/cmnGetChar.h>
#include <cisstCommon/cmnPath.h>
#include <cisstStereoVision.h>
#include <sawOpenNI/svlFilterSourceKinect.h>


int main()
{
    svlInitialize();

    svlStreamManager stream;
    svlFilterSourceKinect kinect;
    svlFilterImageWindow window_rgb;
    svlFilterStreamTypeConverter depth2rgb(svlTypeImageMono16, svlTypeImageRGB);
    svlFilterImageWindow window_depth;

    // Set Kinect configuration file
    cmnPath path;
    path.Add(".");
    std::string configFile = path.Find("SamplesConfig.xml");
    if (configFile == "") {
        std::cerr << "can't find file \"SamplesConfig.xml\" in path: " << path << std::endl;
        exit (-1);
    }
    kinect.SetKinectConfigFile(configFile);

    // Setup Mono16 to RGB converter
    depth2rgb.SetMono16ShiftDown(4);

    // Setup windows
    window_rgb.SetTitle("RGB Image");
    window_depth.SetTitle("Depth Image");

    // Chain filters to trunk
    stream.SetSourceFilter(&kinect);
    kinect.GetOutput("rgb")->Connect(window_rgb.GetInput());
    kinect.GetOutput("depth")->Connect(depth2rgb.GetInput());
    depth2rgb.GetOutput()->Connect(window_depth.GetInput());

    // Initialize and start stream
    if (stream.Play() == SVL_OK) {
        std::cout << "Press any key to stop stream..." << std::endl;
        cmnGetChar();
        std::cout << "Quitting." << std::endl;
    }
    else {
        std::cout << "Error... Quitting." << std::endl;
    }

    // Safely stopping and deconstructing stream before de-allocation
    stream.Release();
    kinect.GetOutput("depth")->Disconnect();

    return 1;
}
