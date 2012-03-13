


import os, time, sys
import numpy as np
import cv, cv2
import pdb
from math import floor
sys.path.append('/Users/colin/code/Kinect-Projects/activityRecognition/')
import SkelPlay as SR

def getFolderTime(folderName):
	# Folder name is of format hr-min-sec
	return (int(folderName[0:2])*60*60 + int(folderName[3:5])*60 + int(folderName[6:8]))
# def getFolderListTimes()

def getFileTime(fileName, fileNames):
	if len(fileNames) < 4:
		return getFolderTime(fileName)
	# fileNames.sort
	start = int(fileNames[0][fileNames[0].find("_")+1:fileNames[0].find(".")])
	end   = int(fileNames[-1][fileNames[-1].find("_")+1:fileNames[-1].find(".")])
	current = int(fileName[fileName.find("_")+1:fileName.find(".")])
	diff = end-start

	return getFolderTime(fileName) + (current-start) / diff

def getFileListTimes(fileNames):
	if len(fileNames) < 4:
		currentTime = getFolderTime(fileNames[0])
		# print currentTime, fileNames
		return [[currentTime, fileNames]]
	start = float(fileNames[0][fileNames[0].find("_")+1:fileNames[0].find(".")])
	end   = float(fileNames[-1][fileNames[-1].find("_")+1:fileNames[-1].find(".")])
	diff = end-start
	times = []

	for f in fileNames:
		current = float(f[f.find("_")+1:f.find(".")])
		currentTime = getFolderTime(f) + float(float(current-start) / diff)
		if len(times) == 0 or times[-1][0] != currentTime:
			currentList = [f]
			times.append([currentTime, currentList])
		else:
			times[-1][1].append(f)

	return times


path = '/Users/colin/data/ICU_7March2012_Head/'
# path = '/Users/colin/data/ICU_7March2012_Foot/'
framerate = 1000;
startTime = 0;
os.chdir(path)
# cv.NamedWindow("RGB")
cv.NamedWindow("DEPTH")

dirNames = os.listdir('.')
dirNames = [x for x in dirNames if (x[0] != '.')] # make sure it's not a hidden file
dirNames.sort()

startDirTime = (getFolderTime(dirNames[0]))
endDirTime = getFolderTime(dirNames[-1])
startTime = time.time() - startTime

currentDirTime = 0
while (time.time() - startTime) < (endDirTime - startDirTime):

	currentTime = (time.time() - startTime) * (framerate / 30)

	# Get current directory
	for i in range(len(dirNames)):
		if currentTime < (getFolderTime(dirNames[i])-startDirTime):
			continue
		else:
			if i+1 < len(dirNames):
				if currentTime < (getFolderTime(dirNames[i+1])-startDirTime):
					currentDirTime = getFolderTime(dirNames[i])
					os.chdir(path+dirNames[i])
					dirNames.pop(i)
					break
				else:
					continue
			else:
				currentDirTime = getFolderTime(dirNames[i])
				os.chdir(path+dirNames[i])
				dirNames.pop(i)
				break


	filenames = os.listdir('.')
	times = getFileListTimes(filenames)
	depthFilename = []
	rgbFilename = []
	skelFilename = []

	tmp = ((currentDirTime - startDirTime))
	tmpMin = int(tmp/60.0)
	tmpSec = int(tmp - tmpMin*60)
	tmp2 = (endDirTime-startDirTime)
	tmpTotMin = int(floor(tmp2/60.0))
	tmpTotSec = int(tmp2 - tmpTotMin*60)
	print "Time: " + str(tmpMin) + " min " + str(tmpSec) + " sec of " + str(tmpTotMin) + " min " + str(tmpTotSec) + " sec"

	# Get current time
	while floor(currentTime) == currentDirTime - startDirTime:
		files = []
				
		#Get closest fileset
		for t in range(len(times)):
			if currentTime > times[t][0] - startDirTime:
				if (t+1)<len(times):
					if currentTime < times[t+1][0] - startDirTime:
						files = times[t][1]
					else:
						continue
				else:
					files = times[t][1]
			else:
				break

		# Get filenames
		if len(files) > 0:
			for i in files:
				if i[-5:] == 'depth':
					depthFilename = i
				if i[-3:] == 'rgb':
					rgbFilename = i
				if i[-4:] == 'skel':
					skelFilename = i

			displayUsers = []
			skels = []
			if len(skelFilename) > 0:
				users = SR.readUserData(skelFilename)
				skels = SR.readSkeletonData(skelFilename)
				for i in users:
					displayUsers.append(users[i]['Img'])

			if len(depthFilename) > 0:
				depthRaw = open(depthFilename, 'rb').read().split()
				depthData = np.array(depthRaw, dtype=int).reshape([480,640])
				depthData = depthData[:,-1:0:-1]

				min_ = np.min(depthData[np.nonzero(depthData)])
				max_ = depthData.max() #/2
				# print min_, max_
				depthData[np.nonzero(depthData)] -= min_
				depthData /= ((max_-min_)/256.0)
				depthData = 256 - depthData

				try:
					for i in displayUsers:
						if i[0] != 0:
							for x in range(-10,10):
								for j in range(-1,1):
									depthData[480 - i[1]+j, 640 - i[0]+x] = 30
							for y in range(-10, 10):
								for j in range(-1,1):
									depthData[480 - i[1]+y, 640 - i[0]+j] = 30
				except:
					print "Error adding cross at", i

				depthData = cv.fromarray(np.array(depthData, dtype=np.uint8))
				# cv.Line(depthData, (0,0), (100,300), [10])
				if skels != []:
					depthData = SR.displaySkeleton_CV(depthData, skels)
				cv2.imshow("DEPTH", np.array(depthData, dtype=np.uint8))

			if len(rgbFilename) > 0:
				imgData = np.fromfile(rgbFilename, dtype=np.uint8)
				imgData = imgData.reshape([480, 640, 3])
				cv2.imshow("RGB", imgData)

			key = cv2.waitKey(33)
			if key > 0:
				break
		currentTime = (time.time() - startTime) * (framerate / 30)				

