/****************************************************************************
 *                                                                           *
 *  OpenNI 1.x Alpha                                                         *
 *  Copyright (C) 2011 PrimeSense Ltd.                                       *
 *                                                                           *
 *  This file is part of OpenNI.                                             *
 *                                                                           *
 *  OpenNI is free software: you can redistribute it and/or modify           *
 *  it under the terms of the GNU Lesser General Public License as published *
 *  by the Free Software Foundation, either version 3 of the License, or     *
 *  (at your option) any later version.                                      *
 *                                                                           *
 *  OpenNI is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the             *
 *  GNU Lesser General Public License for more details.                      *
 *                                                                           *
 *  You should have received a copy of the GNU Lesser General Public License *
 *  along with OpenNI. If not, see <http://www.gnu.org/licenses/>.           *
 *                                                                           *
 ****************************************************************************/
//---------------------------------------------------------------------------
// Includes
//---------------------------------------------------------------------------
#include <XnOpenNI.h>
#include <XnCodecIDs.h>
#include <XnCppWrapper.h>
#include "SceneDrawer.h"
#include <XnPropNames.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <sys/stat.h>
#include <cmath>
using namespace std;

//---------------------------------------------------------------------------
// Globals
//---------------------------------------------------------------------------
xn::Context g_Context;
xn::ScriptNode g_scriptNode;
xn::DepthGenerator g_DepthGenerator;
xn::ImageGenerator g_ImageGenerator;
xn::UserGenerator g_UserGenerator;
xn::Player g_Player;

XnBool g_bNeedPose = FALSE;
XnChar g_strPose[20] = "";
XnBool g_bDrawBackground = TRUE;
XnBool g_bDrawPixels = TRUE;
XnBool g_bDrawSkeleton = TRUE;
XnBool g_bPrintID = TRUE;
XnBool g_bPrintState = TRUE;

#ifndef USE_GLES
#if (XN_PLATFORM == XN_PLATFORM_MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#else
#include "opengles.h"
#endif

#ifdef USE_GLES
static EGLDisplay display = EGL_NO_DISPLAY;
static EGLSurface surface = EGL_NO_SURFACE;
static EGLContext context = EGL_NO_CONTEXT;
#endif

#define GL_WIN_SIZE_X 720
#define GL_WIN_SIZE_Y 480

XnBool g_bPause = false;
XnBool g_bRecord = false;

XnBool g_bQuit = false;

//My Stuff
ofstream skelLogger;    
ofstream depthLogger; 
ofstream rgbLogger; 
const int bufferSize = 30;
// XnDepthPixel depthBuffer[bufferSize];
xn::DepthMetaData depthBuffer[bufferSize];
int currentDepthIndex, frameDiffStep;
int frameInit;
int diffThresh;
time_t timePrev, timeNow;

//#define dir "/Users/colin/data/ICUtest/"
#define dir "/Volumes/ICU/ICUdata/"
		//    string dir = "/Volumes/ICU/testData/";

//---------------------------------------------------------------------------
// Code
//---------------------------------------------------------------------------

void CleanupExit()
{
	g_scriptNode.Release();
	g_DepthGenerator.Release();
    g_ImageGenerator.Release();
	g_UserGenerator.Release();
	g_Player.Release();
	g_Context.Release();
    
	exit (1);
}

#define XN_CALIBRATION_FILE_NAME "UserCalibration.bin"

// Save calibration to file
void SaveCalibration()
{
	XnUserID aUserIDs[20] = {0};
	XnUInt16 nUsers = 20;
	g_UserGenerator.GetUsers(aUserIDs, nUsers);
	for (int i = 0; i < nUsers; ++i)
	{
		// Find a user who is already calibrated
		if (g_UserGenerator.GetSkeletonCap().IsCalibrated(aUserIDs[i]))
		{
			// Save user's calibration to file
			g_UserGenerator.GetSkeletonCap().SaveCalibrationDataToFile(aUserIDs[i], XN_CALIBRATION_FILE_NAME);
			break;
		}
	}
}
// Load calibration from file
void LoadCalibration()
{
	XnUserID aUserIDs[20] = {0};
	XnUInt16 nUsers = 20;
	g_UserGenerator.GetUsers(aUserIDs, nUsers);
	for (int i = 0; i < nUsers; ++i)
	{
		// Find a user who isn't calibrated or currently in pose
		if (g_UserGenerator.GetSkeletonCap().IsCalibrated(aUserIDs[i])) continue;
		if (g_UserGenerator.GetSkeletonCap().IsCalibrating(aUserIDs[i])) continue;
        
		// Load user's calibration from file
		XnStatus rc = g_UserGenerator.GetSkeletonCap().LoadCalibrationDataFromFile(aUserIDs[i], XN_CALIBRATION_FILE_NAME);
		if (rc == XN_STATUS_OK)
		{
			// Make sure state is coherent
			g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(aUserIDs[i]);
			g_UserGenerator.GetSkeletonCap().StartTracking(aUserIDs[i]);
		}
		break;
	}
}

int pixelDifference(const XnDepthPixel* currentDepth, const XnDepthPixel* prevDepth)
{
	int diff = 0;
	for (unsigned int i = 0; i < 307200; i++)
    {
    	if (abs(currentDepth[i] - prevDepth[i]) > 10)
    		diff++;
    }
	return diff;
}

////From: http://pastebin.com/e5kHzs84
//void depth2rgb(const XnDepthPixel* Xn_disparity){
//    int i;
//    //const unsigned short *disparity = Xn_disparity;
//    
//    for (i=0; i<307200; i++) {
//        int pval = depth[Xn_disparity[i]];
//        //printf("%d: %u %d \n",i,*Xn_disparity,depth[*Xn_disparity]);
//        //fflush(stdout);
//        int lb = pval & 0xff;
//        switch (pval>>8) {
//            case 0:
//                depth_data[3*i+0] = 255;
//                depth_data[3*i+1] = 255-lb;
//                depth_data[3*i+2] = 255-lb;
//                break;
//            case 1:
//                depth_data[3*i+0] = 255;
//                depth_data[3*i+1] = lb;
//                depth_data[3*i+2] = 0;
//                break;
//            case 2:
//                depth_data[3*i+0] = 255-lb;
//                depth_data[3*i+1] = 255;
//                depth_data[3*i+2] = 0;
//                break;
//            case 3:
//                depth_data[3*i+0] = 0;
//                depth_data[3*i+1] = 255;
//                depth_data[3*i+2] = lb;
//                break;
//            case 4:
//                depth_data[3*i+0] = 0;
//                depth_data[3*i+1] = 255-lb;
//                depth_data[3*i+2] = 255;
//                break;
//            case 5:
//                depth_data[3*i+0] = 0;
//                depth_data[3*i+1] = 0;
//                depth_data[3*i+2] = 255-lb;
//                break;
//            default:
//                depth_data[3*i+0] = 0;
//                depth_data[3*i+1] = 0;
//                depth_data[3*i+2] = 0;
//                break;
//        }
//    }
//}



// Callback: New user was detected
void XN_CALLBACK_TYPE User_NewUser(xn::UserGenerator& generator, XnUserID nId, void* pCookie)
{
	XnUInt32 epochTime = 0;
	xnOSGetEpochTime(&epochTime);
	printf("%d New User %d\n", epochTime, nId);
    
//    LoadCalibration();
	// New user found
	if (g_bNeedPose)
	{
		g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
	}
	else
	{
		g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
	}
}
// Callback: An existing user was lost
void XN_CALLBACK_TYPE User_LostUser(xn::UserGenerator& generator, XnUserID nId, void* pCookie)
{
	XnUInt32 epochTime = 0;
	xnOSGetEpochTime(&epochTime);
	printf("%d Lost user %d\n", epochTime, nId);	
}
// Callback: Detected a pose
void XN_CALLBACK_TYPE UserPose_PoseDetected(xn::PoseDetectionCapability& capability, const XnChar* strPose, XnUserID nId, void* pCookie)
{
	XnUInt32 epochTime = 0;
	xnOSGetEpochTime(&epochTime);
	printf("%d Pose %s detected for user %d\n", epochTime, strPose, nId);
	g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(nId);
	g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}
// Callback: Started calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(xn::SkeletonCapability& capability, XnUserID nId, void* pCookie)
{
	XnUInt32 epochTime = 0;
	xnOSGetEpochTime(&epochTime);
	printf("%d Calibration started for user %d\n", epochTime, nId);
}
// Callback: Finished calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationComplete(xn::SkeletonCapability& capability, XnUserID nId, XnCalibrationStatus eStatus, void* pCookie)
{
	XnUInt32 epochTime = 0;
	xnOSGetEpochTime(&epochTime);
	if (eStatus == XN_CALIBRATION_STATUS_OK)
	{
		// Calibration succeeded
		printf("%d Calibration complete, start tracking user %d\n", epochTime, nId);		
		g_UserGenerator.GetSkeletonCap().StartTracking(nId);
	}
	else
	{
		// Calibration failed
		printf("%d Calibration failed for user %d\n", epochTime, nId);
        if(eStatus==XN_CALIBRATION_STATUS_MANUAL_ABORT)
        {
            printf("Manual abort occured, stop attempting to calibrate!");
            return;
        }
		if (g_bNeedPose)
		{
			g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
		}
		else
		{
			g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
		}
	}
}


// this function is called each frame
void glutDisplay (void)
{
    
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
	// Setup the OpenGL viewpoint
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
    
	xn::SceneMetaData sceneMD;
	xn::DepthMetaData depthMD;
	g_DepthGenerator.GetMetaData(depthMD);
#ifndef USE_GLES
	glOrtho(0, depthMD.XRes(), depthMD.YRes(), 0, -1.0, 1.0);
#else
	glOrthof(0, depthMD.XRes(), depthMD.YRes(), 0, -1.0, 1.0);
#endif
    
	glDisable(GL_TEXTURE_2D);
    
	if (!g_bPause)
	{
		// Read next available data
//		g_Context.WaitOneUpdateAll(g_UserGenerator);
		g_Context.WaitAndUpdateAll();        
	}


    // // Process the data
    // g_DepthGenerator.GetMetaData(depthMD);
    // g_UserGenerator.GetUserPixels(0, sceneMD);
    // DrawDepthMap(depthMD, sceneMD);
    
    // ------------------------------------------------------
    // My stuff
    // timeNow = clock() / CLOCKS_PER_SEC * 1000.0;
    time(&timeNow);
    
    // Log depth data
    const XnDepthPixel *depthMap = depthMD.Data();

    // depthBuffer[currentDepthIndex] = *depthMap;
    // cout << "Start index: " << currentDepthIndex << endl;
    
    depthBuffer[currentDepthIndex].CopyFrom(depthMD);
    int pixChange;
    if (frameInit == 0)
    {
    	int compareIndex = currentDepthIndex-frameDiffStep;
    	if (compareIndex < 0) compareIndex += bufferSize;

    	const XnDepthPixel *depthPrev = depthBuffer[compareIndex].Data();
    	pixChange = pixelDifference(depthMap, depthPrev);
    	cout << "Pixel diff: " << pixChange << endl;    	
	
		// xn::DepthMetaData dMD2;
		double diff_ = difftime(timeNow, timePrev);
    	// cout << diff_ << endl;
	    if ((diff_ >= 0) && (pixChange > diffThresh || diff_ > 3))
	    {
	    	// timePrev = clock() / CLOCKS_PER_SEC * 1000.0;
	    	time(&timePrev);

	    	// Display depthmap
		    g_UserGenerator.GetUserPixels(0, sceneMD);
		    DrawDepthMap(depthMD, sceneMD);

		    // Get user data
		    XnUserID aUserIDs[20] = {0};
		    XnUInt16 nUsers = 20;
		    g_UserGenerator.GetUsers(aUserIDs, nUsers);
		    XnSkeletonJointPosition joint1;
		    // cout << nUsers << " users" << endl;
		    
		    //Time stamp
		    stringstream ss;
		    //Date w/ seconds
		    time_t rawtime;
		    time(&rawtime);
		    string s_time = ctime(&rawtime);
		    
		    ss << s_time[11] << s_time[12] << "-"; //hr
		    ss << s_time[14] << s_time[15] << "-"; //min
		    ss << s_time[17] << s_time[18]; //sec
		    
		    string folderName = ss.str();
		//     string dir = "/Users/colin/data/ICUtest/";
		// //    string dir = "/Volumes/ICU/testData/";
		    stringstream sDir;
		    sDir << dir;
		    sDir << folderName;
		    mkdir(sDir.str().c_str(), 0755);
		    //Milliseconds
		    long t;
		    t = clock() * CLOCKS_PER_SEC;        
		    ss << "_" << t;        
		    
		    // Get filename prefix
		    stringstream pathPrefix;
		    pathPrefix << dir << folderName << "/";

		    //    string depthFilename = "data/Depth_";  
		    string depthFilename = pathPrefix.str();
		    //    depthFilename.append("Depth_");   
		    depthFilename.append(ss.str().c_str());    
		    depthFilename.append(".depth");
		    cout << depthFilename << endl;
		    depthLogger.open(depthFilename.c_str(), ios::out | ios::binary);
		    for (unsigned int i = 0; i < 307200; i++)
		    {
		        depthLogger << depthMap[i] << " ";
		//        depthLogger << depthMap[i] << endl;
		    }
		    depthLogger.close();
		    
		    // Log rgb data
		    //    xn::ImageMetaData g_imageMD;
		    const XnRGB24Pixel* rgbImg = g_ImageGenerator.GetRGB24ImageMap();
		    //    string rgbFilename = "data/RGB_";  
		    string rgbFilename = pathPrefix.str();      
		    //    rgbFilename.append("RGB_");
		    rgbFilename.append(ss.str().c_str());    
		    rgbFilename.append(".rgb");
		    rgbLogger.open(rgbFilename.c_str(), ios::out | ios::binary);
		    
		    for (unsigned int i = 0; i < 307200; i++)
		    {
		        rgbLogger << rgbImg[i].nRed << rgbImg[i].nGreen << rgbImg[i].nBlue;        
		    }
		    rgbLogger.close();        
		    
		    if (nUsers > 0)
		    {        
		        
		        // Log skeleton file
		        string skelFilename = pathPrefix.str();
		        skelFilename.append(ss.str().c_str());    
		        skelFilename.append(".skel");
		        skelLogger.open(skelFilename.c_str());
		        
		        for (int i=1; i <= nUsers; i++)
		        {
		        	XnPoint3D pt;
					g_UserGenerator.GetCoM (i, pt);
					skelLogger << "User " << i << endl;
					skelLogger << pt.X << " " << pt.Y << " " << pt.Z << endl;         
		        	// cout << pt.X << " " << pt.Y << " " << pt.Z << endl;         
		            
		            skelLogger << "Skeleton " << i << endl;
					
		            for (int j=0; j<25; j++)
		            {
		                XnSkeletonJoint eJoint1 = XnSkeletonJoint(j);
		                g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(i, eJoint1, joint1);
		                //                pt = joint1.position;
		                //            cout << joint1.fConfidence << " pos: " << pt[j].X << " " << pt[j].Y << " " << pt[j].Z << endl;
		                if (joint1.fConfidence >= .5)
		                {
		                    skelLogger << joint1.position.X << " " << joint1.position.Y << " " << joint1.position.Z << endl;                    
		                } else {
		                    skelLogger << 0 << " " << 0 << " " << 0 << endl;                
		                }
		                
		                // skelLogger << joint1.fConfidence << " " << joint1.position.X << " " << joint1.position.Y << " " << joint1.position.Z << endl;                    
		            }
		        }
		        skelLogger.close();
			}
		}
	}
    else 
    {
    	frameInit--;
    }
	currentDepthIndex++;
	if (currentDepthIndex == bufferSize) currentDepthIndex = 0;
	
    // ------------------------------------------------------    
    
#ifndef USE_GLES
	glutSwapBuffers();
#endif
}

#ifndef USE_GLES
void glutIdle (void)
{
	if (g_bQuit) {
		CleanupExit();
	}
    
	// Display the frame
	glutPostRedisplay();
    
}

void glutKeyboard (unsigned char key, int x, int y)
{
	switch (key)
	{
        case 27:
            CleanupExit();
        case 'b':
            // Draw background?
            g_bDrawBackground = !g_bDrawBackground;
            break;
        case 'x':
            // Draw pixels at all?
            g_bDrawPixels = !g_bDrawPixels;
            break;
        case 's':
            // Draw Skeleton?
            g_bDrawSkeleton = !g_bDrawSkeleton;
            break;
        case 'i':
            // Print label?
            g_bPrintID = !g_bPrintID;
            break;
        case 'l':
            // Print ID & state as label, or only ID?
            g_bPrintState = !g_bPrintState;
            break;
        case'p':
            g_bPause = !g_bPause;
            break;
        case 'S':
            SaveCalibration();
            break;
        case 'L':
            LoadCalibration();
            break;
	}
}
void glInit (int * pargc, char ** argv)
{
	glutInit(pargc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(GL_WIN_SIZE_X, GL_WIN_SIZE_Y);
	glutCreateWindow ("User Tracker Viewer");
	//glutFullScreen();
	glutSetCursor(GLUT_CURSOR_NONE);
    
	glutKeyboardFunc(glutKeyboard);
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutIdle);
    
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
    
	glEnableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}
#endif // USE_GLES

#define SAMPLE_XML_PATH "../SamplesConfig.xml"

#define CHECK_RC(nRetVal, what)										\
if (nRetVal != XN_STATUS_OK)									\
{																\
printf("%s failed: %s\n", what, xnGetStatusString(nRetVal));\
return nRetVal;												\
}

int main(int argc, char **argv)
{
	if (argc > 1)
		diffThresh = atoi(argv[1]);
	else
		diffThresh = 102000;
	currentDepthIndex = 0;
	frameDiffStep = 10;
	frameInit = frameDiffStep+1;	
	time(&timePrev);
	XnStatus nRetVal = XN_STATUS_OK;
    
	if (argc > 2) // if (argc > 1)
	{
		nRetVal = g_Context.Init();
		CHECK_RC(nRetVal, "Init");
		nRetVal = g_Context.OpenFileRecording(argv[1], g_Player);
		if (nRetVal != XN_STATUS_OK)
		{
			printf("Can't open recording %s: %s\n", argv[1], xnGetStatusString(nRetVal));
			return 1;
		}
	}
	else
	{
		xn::EnumerationErrors errors;
		nRetVal = g_Context.InitFromXmlFile(SAMPLE_XML_PATH, g_scriptNode, &errors);
		if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
		{
			XnChar strError[1024];
			errors.ToString(strError, 1024);
			printf("%s\n", strError);
			return (nRetVal);
		}
		else if (nRetVal != XN_STATUS_OK)
		{
			printf("Open failed: %s\n", xnGetStatusString(nRetVal));
			return (nRetVal);
		}
	}

    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_IMAGE, g_ImageGenerator);
    if (nRetVal != XN_STATUS_OK)
    {
        printf("Error setting up image");
    }
	nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
	if (nRetVal != XN_STATUS_OK)
	{
		printf("No depth generator found. Using a default one...");
		xn::MockDepthGenerator mockDepth;
		nRetVal = mockDepth.Create(g_Context);
		CHECK_RC(nRetVal, "Create mock depth");
        
		// set some defaults
		XnMapOutputMode defaultMode;
		defaultMode.nXRes = 320;
		defaultMode.nYRes = 240;
		defaultMode.nFPS = 10; // normally 30
		nRetVal = mockDepth.SetMapOutputMode(defaultMode);
		CHECK_RC(nRetVal, "set default mode");
        
		// set FOV
		XnFieldOfView fov;
		fov.fHFOV = 1.0225999419141749;
		fov.fVFOV = 0.79661567681716894;
		nRetVal = mockDepth.SetGeneralProperty(XN_PROP_FIELD_OF_VIEW, sizeof(fov), &fov);
		CHECK_RC(nRetVal, "set FOV");
        
		XnUInt32 nDataSize = defaultMode.nXRes * defaultMode.nYRes * sizeof(XnDepthPixel);
		XnDepthPixel* pData = (XnDepthPixel*)xnOSCallocAligned(nDataSize, 1, XN_DEFAULT_MEM_ALIGN);
        
		nRetVal = mockDepth.SetData(1, 0, nDataSize, pData);
		CHECK_RC(nRetVal, "set empty depth map");
        
		g_DepthGenerator = mockDepth;
	}
    
	nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_USER, g_UserGenerator);
	if (nRetVal != XN_STATUS_OK)
	{
		nRetVal = g_UserGenerator.Create(g_Context);
		CHECK_RC(nRetVal, "Find user generator");
	}
    
	XnCallbackHandle hUserCallbacks, hCalibrationStart, hCalibrationComplete, hPoseDetected, hCalibrationInProgress, hPoseInProgress;
	if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON))
	{
		printf("Supplied user generator doesn't support skeleton\n");
		return 1;
	}
	nRetVal = g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, hUserCallbacks);
	CHECK_RC(nRetVal, "Register to user callbacks");
	nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationStart(UserCalibration_CalibrationStart, NULL, hCalibrationStart);
	CHECK_RC(nRetVal, "Register to calibration start");
	nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationComplete(UserCalibration_CalibrationComplete, NULL, hCalibrationComplete);
	CHECK_RC(nRetVal, "Register to calibration complete");
    
	if (g_UserGenerator.GetSkeletonCap().NeedPoseForCalibration())
	{
		g_bNeedPose = TRUE;
		if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_POSE_DETECTION))
		{
			printf("Pose required, but not supported\n");
			return 1;
		}
		nRetVal = g_UserGenerator.GetPoseDetectionCap().RegisterToPoseDetected(UserPose_PoseDetected, NULL, hPoseDetected);
		CHECK_RC(nRetVal, "Register to Pose Detected");
		g_UserGenerator.GetSkeletonCap().GetCalibrationPose(g_strPose);
	}
    
	g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);
    
	nRetVal = g_UserGenerator.GetSkeletonCap().RegisterToCalibrationInProgress(MyCalibrationInProgress, NULL, hCalibrationInProgress);
	CHECK_RC(nRetVal, "Register to calibration in progress");
    
	nRetVal = g_UserGenerator.GetPoseDetectionCap().RegisterToPoseInProgress(MyPoseInProgress, NULL, hPoseInProgress);
	CHECK_RC(nRetVal, "Register to pose in progress");
    
	nRetVal = g_Context.StartGeneratingAll();
	CHECK_RC(nRetVal, "StartGenerating");
    
#ifndef USE_GLES
	glInit(&argc, argv);
	glutMainLoop();
#else
	if (!opengles_init(GL_WIN_SIZE_X, GL_WIN_SIZE_Y, &display, &surface, &context))
	{
		printf("Error initializing opengles\n");
		CleanupExit();
	}
    
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
    
//    cvNamedWindow("RGB", CV_WINDOW_AUTOSIZE);
//    cvNamedWindow("Depth", CV_WINDOW_AUTOSIZE);
//    IplImage *rgbimg = cvCreateImageHeader(cvSize(640,480), 8, 3);
//    IplImage *depthimg = cvCreateImageHeader(cvSize(640,480), 8, 3); 
    
    
	while (!g_bQuit)
	{
		glutDisplay();
		eglSwapBuffers(display, surface);        
    }
	opengles_shutdown(display, surface, context);
    
	CleanupExit();
#endif
}
