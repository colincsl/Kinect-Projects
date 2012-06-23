
import os, time, sys
import numpy as np
import cv, cv2
import scipy.ndimage as nd
import pdb
from math import floor
sys.path.append('/Users/colin/code/Kinect-Projects/activityRecognition/')
from icuReader import ICUReader
from peopleTracker import Tracker
from SkelPlay import *
from backgroundSubtract import *
from featureExtraction import *
from copy import deepcopy

import dijkstras


# Inter-person vector comparisons
# Always have n by 5 vector
def orientationComparison(vecs1, direc=2, size_=5):
	# Find the projection of each person's vector towards each other
	# pdb.set_trace()
	vecCompare = np.zeros([len(vecs1), size_])
	for i in xrange(len(vecs1)):
		for j in xrange(min(len(vecs1), size_)):
			if i == j:#j < i:
				continue
			else:
				vecCompare[i,j] = np.abs(np.dot(vecs1[i][direc], vecs1[j][direc]))
	vecCompare = -1.0*np.sort(-vecCompare, axis=1)
	return vecCompare



#------------------Init-----------------------
#----Get mean images for bg subtraction-------
# path = '/Users/colin/data/ICU_7March2012_Head/'
# path2 = '/Users/colin/data/ICU_7May2012_Wide/'
framerate = 30;
path = '/Users/colin/data/ICU_7May2012_Close/'
startTime = 1600; #close
path = '/Users/colin/data/ICU_7May2012_Wide/'
startTime = 8000; #wide
#---------------------------------------------
startTime2 = startTime+12;
reader1 = ICUReader(path, framerate, startTime, cameraNumber=0, viz=0, vizSkel=0, skelsEnabled=0)
# reader2 = ICUReader(path2, framerate, startTime2, cameraNumber=1, viz=0, vizSkel=0, skelsEnabled=0)

depthImgs1 = []
depthImgs2 = []
depthStackInd1 = 0
depthStackInd2 = 0
for i in xrange(10):
	reader1.run()
	depthImgs1.append(reader1.depthDataRaw)
	# reader2.run()
	# depthImgs2.append(reader2.depthDataRaw)

depthImgs1 = np.dstack(depthImgs1)
# depthImgs2 = np.dstack(depthImgs2)

mean1 = getMeanImage(depthImgs1)
# mean2 = getMeanImage(depthImgs2)
m1 = constrain(mean1, 500, 4000)
# m2 = constrain(mean2, 500, 4500)

#-----------------------------------------
#--------------MAIN-----------------------
#-----------------------------------------

framerate = 100
startTime = 6902#6900  #OLD:#2000 #2x procedure: 6900, 7200
# startTime = 600#6900  #OLD:#2000 #2x procedure: 6900, 7200
# startTime = 1350#2000
startTime2 = startTime+12
serial = True
if 0: #Serial
	reader1 = ICUReader(path, framerate, startTime, cameraNumber=0, viz=0, vizSkel=0, skelsEnabled=0, serial=1)
	# reader2 = ICUReader(path2, framerate, startTime2, cameraNumber=1, viz=0, vizSkel=0, skelsEnabled=0, serial=1)
else: #Real-time
	reader1 = ICUReader(path, framerate, startTime, cameraNumber=0, viz=0, vizSkel=0, skelsEnabled=0, serial=0)
	# reader2 = ICUReader(path2, framerate, startTime2, cameraNumber=1, viz=0, vizSkel=0, skelsEnabled=0, serial=0)

vizWin = 1
if vizWin:
	cv.NamedWindow("a")
	# cv.NamedWindow("a_seg")
	# cv.NamedWindow("b")
	# cv.NamedWindow("b_seg")

# from multiprocessing import Pool, Queue, Process
### Can't pass numpy arrays into processes! Must use ctypes
## See numpy-sharedmem

dir_ = '/Users/colin/code/Kinect-Projects/activityRecognition/'
tracker1 = Tracker('1', dir_)
tracker2 = Tracker('2', dir_)
featureExt1 = Features(['basis']) #feature extractor
# featureExt1 = Features(['basis', 'viz']) #feature extractor
# featureExt1.addTouchEvent([[-5, -5000, -5000], [5, 5000, 5000]])
###### problem between xyz/uvw??
#Old footage
# featureExt1.addTouchEvent([250, -200, 1000], 350)
# featureExt1.addTouchEvent([-175, -150, 1150], 300)
#New Footage
featureExt1.addTouchEvent([-146.83551756, -16.10465379, 3475.0], 350)
featureExt1.addTouchEvent([-141.13408991, -251.45427198, 2194.0], 300)
# featureExt1.addTouchEvent([0, 0, 2100], 200)
featureExt2 = Features(['basis', 'viz']) #feature extractor

# for i in xrange(1):
# while(1):
histVecs = []
start = time.time()
# while (len(reader1.allPaths) > 0):
# while(1):
if 1:

	if 1:
		try:
		# if 1:
			tStart = time.time()
			reader1.run()	
			d1 = reader1.depthDataRaw
			d1c = constrain(d1, 500, 4000)
			# diff = m1 - d1c
			# diffDraw1 = d1c*(diff > 50)*(diff < 225)			
			diff = np.array(m1, dtype=int16) - np.array(d1c, dtype=int16)
			diffDraw1 = d1c*(diff > 20)			
			out1, objects1, labelInds1 = extractPeople_2(diffDraw1)
			if len(labelInds1) > 0:
				d1, com1, vecs1, touched1 = featureExt1.run(d1, out1, objects1, labelInds1)
				ornCompare = orientationComparison(vecs1)
				com1_xyz = featureExt1.coms_xyz
				t = reader1.timeMin*60 + reader1.timeSec
				ids = tracker1.run(com1_xyz, objects1, t, reader1.depthFilename, touched1, vecs1, ornCompare)
				# highlight = generateKeypoints(objects1, labelInds1, out1, d1, com1, featureExt1)

				# for i in xrange(len(ids)):
				# 	# pdb.set_trace()
				# 	v = ids[i]
				# 	if len(histVecs) <= v:
				# 		histVecs.append([])
				# 	histVecs[v].append(vecs1[i])

				# print "People: ", ids

			# extremaInds = []
			# for i in xrange(len(labelInds1)):
			# 	try:
			# 		extrema = getExtrema(objects1, labelInds1, out1, d1, com1, featureExt1, i)
			# 		extremaInds.append(extrema)
			# 	except:
			# 		print "Error getting extrema"

			tEnd = time.time()
			# print "Time 1: ", tEnd - tStart

			if vizWin:
				out1b = np.zeros_like(out1, dtype=np.uint8)+255
				out1b = np.dstack([out1b, out1b, out1b])
				d1c = constrain(d1, 500, 4000)
				d1c = np.dstack([d1c, d1c, d1c])
				# Draw binary sensors
				if 0:
					for i in featureExt1.touchAreas:
						center = i[0]
						center = world2depth(np.array([center]))
						r = int(i[1]*0.27)
						cv2.circle(d1c, (center[1][0],center[0][0]), r, [0,100, 100], thickness=2)

				for i in xrange(len(labelInds1)):
					# out1[out1==labelInds1[i]] = (ids[i]+1)*50
					if 1:
						d1c[out1==labelInds1[i], ids[i]%3] = (ids[i]+1)*50
						d1c[out1==labelInds1[i], (ids[i]+1)%3] = 0
						d1c[out1==labelInds1[i], (ids[i]+2)%3] = 0
					if 0:
						# print "t", touched1
						d1c[out1==labelInds1[i], 1] = 75 * (ids[i]%3 == 0)
						d1c[out1==labelInds1[i], 2] = 75 * (ids[i]%3 == 1)
						d1c[out1==labelInds1[i], 0] = 75 * (ids[i]%3 == 2)

						for j in range(len(touched1)):
							t = [labelInds1[x] for x in touched1[j]]

							if labelInds1[i] in t:
								d1c[out1==labelInds1[i], 1] = 255 * (ids[i]%3 == 0)
								d1c[out1==labelInds1[i], 2] = 255 * (ids[i]%3 == 1)
								d1c[out1==labelInds1[i], 0] = 255 * (ids[i]%3 == 2)
								# Draw bigger radius
								center = featureExt1.touchAreas[j][0]
								center = world2depth(np.array([center]))
								r = int(featureExt1.touchAreas[j][1]*.1)
								cv2.circle(d1c, (center[1][0],center[0][0]), r, [0,150, 150], thickness=4)

				if len(labelInds1) > 0:
					d1c[highlight[0]-5:highlight[0]+6, highlight[1]-5:highlight[1]+6,:] = 255

				# for exSet in extremaInds:
				# 	for ex in exSet:
				# 		d1c[ex[0]-2:ex[0]+2, ex[1]-2:ex[1]+2] = 255

				cv2.imshow("a", d1c)
				out1 *= 10 * (out1>0)
				out1s = np.array(np.dstack([out1*10, out1*11, out1*12]), dtype=uint8)
				# cv2.imshow("a_seg", out1b)


		except:
			print 'Error in camera 1'

	ret = cv2.waitKey(1)
	# if ret > 0:
		# break

	print "Time: ", (reader1.currentDirTime - reader1.startDirTime) , ", left: " , len(reader1.allPaths)

tracker1.finalize()
end = time.time()
print end - start





	## Adapt background
	# if len(objects1) == 0:
	# 	depthImgs1[:,:,depthStackInd1] = d1
	# 	mean1 = getMeanImage(depthImgs1)
	# 	m1 = constrain(mean1, 500, 4000)
	# 	depthStackInd1 += 1
	# 	if depthStackInd1 == 5:
	# 		depthStackInd1 = 0
	# if len(objects2) == 0:
	# 	depthImgs2[:,:,depthStackInd2] = d2
	# 	mean2 = getMeanImage(depthImgs2)
	# 	m2 = constrain(mean2, 500, 6000)
	# 	depthStackInd2 += 1
	# 	if depthStackInd2 == 5:
	# 		depthStackInd2 = 0

	
	# out1, objects1 = extractPeople(diffDraw1)
	# out2, objs2 = extractPeople(diffDraw2)	

	#######
	# tStart = time.time()	
	# # Camera 1
	# d1 = reader1.depthDataRaw
	# diff1 = d1*(np.abs(d1 - mean1) > 200)
	# diffDraw1 = constrain(diff1, 500, 4000)
	# d2 = reader2.depthDataRaw
	# diff2 = d2*(np.abs(d2 - mean2) > 200)
	# diffDraw2 = constrain(diff2, 500, 6000)
	
	# # pool = Pool(processes = 4)
	# # res1 = pool.map_async(extractPeople, (diffDraw1,)
	# # results = pool.map_async(extractPeople, [diffDraw1, diffDraw2])
	# # out1, objects1 = extractPeople(diffDraw1)
	# # out2, objs2 = extractPeople(diffDraw2)
	# pool.close()
	# pool.join()
	# # x = res1.get()
	# # x, y = results.get()

	# # dSeg = constrain(out1, out1.min(), out1.max())
	# # dSeg2 = constrain(out2, out2.min(), out2.max())

	# tEnd = time.time()
	# print "Time 2: ", tEnd - tStart	

#-----------Segment attempt 2. Convolutions------------------------

# generic_filter(input, function, size=None, footprint=None, output=None, mode='reflect', cval=0.0, origin=0, extra_arguments=(), extra_keywords={})
# filter_ = np.array([[1.0, 2.0], [3.0, 4.0]])
# footprint = np.array([[1,0],[0,1]])
# func = lambda x: np.diff(x) < 10
# x = nd.generic_filter(im, func, size=(1,2)) # x-axis
# y = nd.generic_filter(im, func, size=(2,1)) # x-axis
# imshow(x*im)
# imshow(y*im)

# func = lambda x: np.max(np.diff(x)) < 30
# grad = nd.generic_filter(im, func, size=(2,2)) # x-axis

#-----------Keypoints------------------------
cd '/Users/colin/code/Kinect-Projects/activityRecognition/'
np.savez('tmpPerson_close.npz', {'objects':objects1, 'labels':labelInds1, 'out':out1, 'd':d1, 'com':com1, 'features':featureExt1})


# saved = np.load('tmpPerson.npz')['arr_0'].tolist()
saved = np.load('tmpPerson_close.npz')['arr_0'].tolist()
objects1 = saved['objects']; labelInds1=saved['labels']; out1=saved['out']; d1=saved['d']; com1=saved['com'];featureExt1=saved['features']

def generateKeypoints(objects1, labelInds1, out1, d1, com1, featureExt1):

	timeStart = time.time()
	viz = 0

	objects = objects1
	labelInds = labelInds1
	labelInds = [10, 15]
	labelInds = [1,3]

	out = out1
	d=d1
	com = com1
	ind = 0
	mask = out[objects[ind]]==labelInds[ind]
	imgBox = d1[objects[ind]]
	# mask_erode = nd.binary_erosion(out[objects[ind]]==labelInds[ind], iterations=1)
	# mask_erode = nd.binary_closing(out[objects[ind]]==labelInds[ind], iterations=5)
	mask_erode = nd.binary_dilation(out[objects[ind]]==labelInds[ind], iterations=2)
	objTmp = np.array(d[objects[ind]])#, dtype=np.uint16)

	# cv.PyrDown(cv.fromarray(objTmp), cv.fromarray(objTmpLow))
	# objTmp = objTmpLow

	obj2Size = np.shape(objTmp)
	x = objects[ind][0].start # down
	y = objects[ind][1].start # right
	c = np.array([com[ind][0] - x, com[ind][1] - y])
	current = [c[0], c[1]]

	tmp1 = np.nonzero(mask>0)
	t = argmax(tmp1[0])
	current = [tmp1[0][t]-5, tmp1[1][t]]

	xyz = featureExt1.xyz[ind]
	trail = []
	allTrails = []
	singleTrail = set()

	posMat = np.zeros([obj2Size[0], obj2Size[1], 3], dtype=float)
	tmp = np.nonzero(mask_erode)
	v = np.vstack([tmp[1]+y, tmp[0]+x, imgBox[tmp]]).T
	allPos = depth2world(v)
	posMat[v[:,1]-x,v[:,0]-y] = allPos

	objTmp = posMat
	dists2 = np.empty([obj2Size[0]-2,obj2Size[1]-2,4], dtype=int16)
	dists2[:,:,0] = 10*np.sum(np.abs(objTmp[1:-1, 1:-1] - objTmp[0:-2, 1:-1]), 2)#up
	dists2[:,:,1] = 10*np.sum(np.abs(objTmp[1:-1, 1:-1] - objTmp[2:, 1:-1]), 2)#down
	dists2[:,:,2] = 10*np.sum(np.abs(objTmp[1:-1, 1:-1] - objTmp[1:-1, 2:]), 2)#right
	dists2[:,:,3] = 10*np.sum(np.abs(objTmp[1:-1, 1:-1] - objTmp[1:-1, 0:-2]), 2)#left
	dists2[-mask_erode[1:-1,1:-1]] = 0#32000
	dists2 = np.abs(dists2)

	# dists2copy = deepcopy(dists2)

	# extrema = []
	extrema = [current]
	t2 = time.time()
	for i in xrange(10):
		dists2Tot = np.zeros([obj2Size[0],obj2Size[1]], dtype=int16)+32000		
		maxDists = np.max(dists2, 2)
		distThresh = 500
		outline = np.nonzero(maxDists>distThresh)
		mask[outline[0]+1, outline[1]+1] = 0

		# dists2Tot[dists2Tot > 0] = 32000
		# dists2Tot[-mask] = 15000
		dists2Tot[-mask_erode] = 1#15000
		dists2Tot[current[0], current[1]] = 0

		visitMat = np.zeros_like(dists2Tot, dtype=uint8)
		# visitMat[-mask] = 255		
		visitMat[-mask_erode] = 255

		for j in singleTrail:
			dists2Tot[j[0], j[1]] = 0
			dists2[j[0],j[1],:] = 0
			if 0 < (j[0]+1) < obj2Size[0]-2 and 0 < (j[1]+1) < obj2Size[1]-2:
				dists2[j[0]+1,j[1],0] = 0
				dists2[j[0]-1,j[1],1] = 0
				dists2[j[0],j[1]+1,2] = 0
				dists2[j[0],j[1]-1,3] = 0

		trail = dijkstrasGraph.graphDijkstras(dists2Tot, visitMat, dists2, current)

		# dists2Tot *= mask_erode
		# dists2Tot *= mask
		# dists2Tot[1:,:] *= ((dists2Tot[1:,:] - dists2Tot[0:-1,:]) < 1000)
		# dists2Tot[:-1,:] *= ((dists2Tot[:-1,:] - dists2Tot[1:,:]) < 1000)
		# dists2Tot[:,1:] *= ((dists2Tot[:,1:] - dists2Tot[:,:-1]) < 1000)
		# dists2Tot[:,:-1] *= ((dists2Tot[:,:-1] - dists2Tot[:,1:]) < 1000)




		# maxInd = np.argmax(dists2Tot*(dists2Tot<30000))
		# maxInd = np.unravel_index(maxInd, dists2Tot.shape)
		# maxInd = (trail[-1][0],trail[-1][1])
		maxInd = (trail[0][0],trail[0][1])

		allTrails.append(trail)
		for j in trail:
			if j[0] > 0 and j[1] > 0:
				singleTrail.add((j[0], j[1]))

		# extrema.append(trail[-1])
		# current = [extrema[-1]]
		extrema.append([maxInd[0], maxInd[1]])
		current = [maxInd[0], maxInd[1]]
		# print current
	print "t1: ", time.time() - t2

	trailLens = []
	for i in allTrails:
		trailLens.append(len(i))
	maxLen = np.argmax(trailLens)

	if allTrails != []:
		for trails_i in allTrails:
			for i in trails_i:
				if i[0] > 0 and i[1] > 0:
					dists2Tot[i[0], i[1]] = 32000
	for i in extrema:
		dists2Tot[i[0]-3:i[0]+4, i[1]-3:i[1]+4] = 10000#799

	# dists2Tot[extrema[maxLen][0]-3:extrema[maxLen][0]+4, extrema[maxLen][1]-3:extrema[maxLen][1]+4] = 400

	if viz:
		figure(1); imshow(dists2Tot*(dists2Tot <= 32000)*(dists2Tot > 0))

	''' ----------- '''

	# distance = dijkstrasGraph.AStar([100,100],[110,100], visitMat, dists2)

	# extrema = [x for x in extrema if x[1] < objTmp.shape[0]-5]
	# extrema = [x for x,i in zip(extrema, range(len(extrema))) if np.sum(nodeDists[:,i]) > 0]
	
	#--------------------------------------
	''' Find distances on the manifold '''
	# nodeDists = np.ones([len(extrema), len(extrema)], dtype=uint16)
	# nodeClosest = np.zeros([len(extrema)], dtype=int8)-1

	# extremaDists = []
	# allDistsMats = []
	# figure(2)
	# current = [tmp1[0][t]-5, tmp1[1][t]]
	# for i in range(len(extrema)):
	# 	current = extrema[i]
	# 	dists2Tot = np.zeros([obj2Size[0],obj2Size[1]], dtype=int16)+32000		
	# 	maxDists = np.max(dists2, 2)
	# 	distThresh = 30
	# 	outline = np.nonzero(maxDists>distThresh)
	# 	mask[outline[0]+1, outline[1]+1] = 0

	# 	# dists2Tot[dists2Tot > 0] = 32000
	# 	# dists2Tot[-mask] = 15000
	# 	dists2Tot[-mask_erode] = 1 #15000
	# 	dists2Tot[current[0], current[1]] = 0

	# 	visitMat = np.zeros_like(dists2Tot, dtype=uint8)
	# 	# visitMat[-mask] = 255
	# 	visitMat[-mask_erode] = 255

	# 	dijkstrasGraph.graphDijkstras(dists2Tot, visitMat, dists2copy, extrema[i])
	# 	dists2Tot *= mask_erode
	# 	exDist = []
	# 	allDistsMats.append(deepcopy(dists2Tot))

	# 	for ex in extrema:
	# 		exDist.append(dists2Tot[ex[0],ex[1]])
	# 	for ex_i in xrange(len(extrema)):
	# 		nodeDists[i, ex_i] = dists2Tot[extrema[ex_i][0],extrema[ex_i][1]]
	# 	sortedDists = sort(exDist)[0:4]
	# 	sortedInds = argsort(exDist)[0:4]
	# 	nodeClosest[i] = sortedInds[0]


	# 	plot(extrema[i][1],-extrema[i][0], 'ro')
	# 	for j in range(4):
	# 		plot([extrema[i][1],extrema[sortedInds[j]][1]],[-extrema[i][0],-extrema[sortedInds[j]][0]], 'g-')
	# 		# axis([0, 640, 0, 480])
	# 	# show()
	# 	# print extrema[i], sortedInds
	# 	extremaDists.append(exDist)
	# axis('equal')

	# nodeDists = np.ones([len(extrema), len(extrema)], dtype=uint16)
	# nodeClosest = np.zeros([len(extrema)], dtype=int8)-1

	# for i in range(len(extrema)):
	# 	for ex in extrema:
	# 		exDist.append(allDistsMats[i][ex[0],ex[1]])
	# 	for ex_i in xrange(len(extrema)):
	# 		nodeDists[i, ex_i] = allDistsMats[i][extrema[ex_i][0],extrema[ex_i][1]]


	# extrema = [x for x,i in zip(extrema, range(len(extrema))) if np.sum(nodeDists[:,i]) > 0]

	# def minSpanTree():

	#--------------------------------------
	# mstList = [0]
	# currentNode = 0
	# nodes = range(1,len(extrema[1:])+1)
	# while nodes != []:
	# 	minDist = np.inf
	# 	closestNode = -1
	# 	for curr in mstList:
	# 		for n in nodes:
	# 			if nodeDists[n,curr] < minDist and 0 < nodeDists[n,curr] < 100000 and 0 < nodeDists[curr,n] < 100000:
	# 				minDist = nodeDists[n,curr]
	# 				closestNode = n
	# 	if closestNode >= 0:
	# 		currentNode = closestNode
	# 		nodes.remove(currentNode)
	# 		mstList.append(currentNode)
	# 		print minDist
	# 	else:
	# 		print 'Error'
	# 		break	


	# figure(2)
	# for i in xrange(len(extrema)):
	# 	plot(extrema[i][1],-extrema[i][0], 'ro')
	# for j in xrange(1,len(mstList)):
	# 	plot([extrema[mstList[j]][1],extrema[mstList[j-1]][1]],[-extrema[mstList[j]][0],-extrema[mstList[j-1]][0]], 'b-')
	# axis('equal')

	# ''' Clustering test '''
	# # from sklearn.cluster import AffinityPropagation as AP

	# D = nodeDists
	# S = 1 - (D / np.max(D))

	# # from sklearn.cluster import DBSCAN
	# # from sklearn.cluster import MeanShift as MS

	# # clust = DBSCAN().fit(S)
	# # clust = MS().fit(S)	
	# clust = AP().fit(S)

	# labels = clust.labels_
	# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	# print n_clusters, "cluster"
	# colors = 'rgbyk'*5
	# figure(3)
	# for i in xrange(len(extrema)):
	# 	plot(extrema[i][1],-extrema[i][0], 'o', color=colors[labels[i]])
	# axis('equal')

	# interpolation='nearest'


	''' Put gaussian around each interest point using RANSAC'''
	# import scipy.stats as sciStats
	# sampleCount = 50
	# modelCenter = np.mean(xyz, 0)
	# modelStd = [200, 100, 50]

	# Convolution of Guassians
	# def gaussConvolution(m1, )
	# handSize = 0.15
	# headSize = 0.25
	# bodySize = 0.50

	# humanStructure = [
	# 	    head
	# 	     |
	# 	  *torso*
	#    sh  |   sh
	#  arm   |	  arm
	#hand    |		hand
	#       
	head = .15
	lArm = .25
	lHand = .25
	rArm = .25
	rHand = .25

	# torso: [head, lArm, rArm]

	# humanStructure = [
	# [0,0],]

	handRatio = .4
	headRatio = .6
	bodyRatio = .95
	ratios = [handRatio, headRatio, bodyRatio]


	oldCenter = extrema[1]
	# oldCenter = np.array([0,0,0])
	oldStd = np.array([9999,9999,9999])
	tmpPts = []
	x = objects[ind][0].start # down
	xMax = objects[ind][0].stop-x
	y = objects[ind][1].start # right
	yMax = objects[ind][1].stop-y

	extRatios = []
	newCenters = []
	boxThresh = 20
	for i in xrange(1, len(extrema)):
		oldCenter = extrema[i]
		for j in range(10):
			x_l=min(max(oldCenter[0]-boxThresh, 0),xMax); x_r=min(max(oldCenter[0]+boxThresh+1, 0),xMax)
			y_l=min(max(oldCenter[1]-boxThresh, 0),yMax); y_r=min(max(oldCenter[1]+boxThresh+1, 0),yMax)
			tmpBox = posMat[x_l:x_r, y_l:y_r]
			tmpInds = np.nonzero(tmpBox[:,:,0])
			tmpPts = tmpBox[tmpInds]
			tmpMeanXYZ = np.mean(tmpPts, 0)
			tmpMeanInd = world2depth(np.array([tmpMeanXYZ])).T[0]

			# _,_,vT = svd(tmpPts-tmpMeanXYZ, full_matrices=0)
			_,_,vT = svd(tmpPts[:,0:2]-tmpMeanXYZ[0:2], full_matrices=0)
			vT = vT.T
			ang = np.arccos(vT[0,0])*180/3.14+90 # in degrees
			tmpBoxRot = nd.rotate(tmpBox[:,:,2], ang)
			# newPts = np.asarray(np.asmatrix(tmpPts-tmpMeanXYZ)*np.asmatrix(vT))+tmpMeanXYZ
			newPts = np.asarray(np.asmatrix(tmpPts-tmpMeanXYZ)*np.eye(3))+tmpMeanXYZ
			# newInd = world2depth(np.array([newPts[:,1],newPts[:,0],newPts[:,2]]).T).T
			# newInd = np.array([x_ for x_ in newInd if x_[0] >= 0 and x_[1] >= 0])
			# posMat[x_l:x_r, y_l:y_r] = 0
			# posMat[newInd[:,1]-x,newInd[:,0]-y] = newInd[:,2]

			# tmpBox[:,:,2] = 0
			# tmpBox[tmpInds[0],tmpInds[1],2] = newPts[:,2]

			# tmpStd = sort(np.asarray(np.std(newPts, 0)))
			tmpStd = np.asarray(np.std(newPts, 0))

			oldCenter = [tmpMeanInd[1]-x, tmpMeanInd[0]-y]

		if viz:
			figure(4); 
			subplot(2,ceil(len(extrema)/2),i)#+1
			imshow(tmpBoxRot)
			figure(5); 
			subplot(2,ceil(len(extrema)/2),i)#+1
			imshow(tmpBox[:,:,2])		
			title(oldCenter)
			print "Arm=gr, Head=or, Body=red"
		
		# print tmpStd, min(tmpStd[0]/tmpStd[1], tmpStd[1]/tmpStd[0])
		extRatios.append(min(tmpStd[0]/tmpStd[1], tmpStd[0]/tmpStd[1]))
		newCenters.append(oldCenter)

	newErode = np.array(mask_erode>0, dtype=int)
	for i in range(len(newCenters)):
		label = argmin((ratios-extRatios[i])**2)
		# print label
		pos = newCenters[i]
		pos = extrema[i]
		newErode[pos[0]-3:pos[0]+4, pos[1]-3:pos[1]+4] = 2+label

	if viz:
		figure(1); imshow(newErode)


	print "Time: ", time.time() - timeStart

		# if np.sqrt(np.sum((newStd-modelStd)**2, 0)) < np.sqrt(np.sum((oldStd-modelStd)**2, 0)):
		# 	oldCenter = deepcopy( newCenter)
		# 	oldStd = deepcopy(newStd)

		# print 'n', newCenter, newStd
	




# # LU Arm
# # Sample points not in torso. Fit line
# for i in xrange(100):
# 	# current = world2depth(np.array([oldCenter])).T[0]
# 	# current = [current[0]-x, current[1]-y]
# 	# visitMat = np.zeros_like(dists2Tot, dtype=uint8)
# 	# visitMat[-mask_erode] = 255
# 	# dists2Tot[dists2Tot > 0] = 9999
# 	# dists2Tot[-mask_erode] = 15000
# 	# dists2Tot[current[0], current[1]] = 0
# 	# dijkstrasGraph.dijkstras(dists2Tot, visitMat, dists2, current)
	
# 	tmpPts = xyz[randint(0, xyz.shape[0], sampleCount)]
# 	if i==0:
# 		oldCenter = np.mean(tmpPts, 0)
# 	tmpPts -= oldCenter
# 	# Align with PCA?
# 	keepPts = np.nonzero(np.sum(np.less_equal(tmpPts, var),1)==3)
# 	oldCenter = np.mean(xyz[keepPts], 0)
# 	print oldCenter
	
# 	# scipy.stats.norm





	# for rad in range(1, 65):
	# 	xRange = slice(c[0]-rad,c[0]+rad)
	# 	yRange = slice(c[1]-rad,c[1]+rad)
	# 	dists2Tot[xRange, yRange] = np.min([dists2Tot[xRange, yRange],
	# 						dists2[xRange, yRange,0] + dists2Tot[c[0]-rad-1:c[0]+rad-1, c[1]-rad:c[1]+rad], #up
	# 						dists2[xRange, yRange,1] + dists2Tot[c[0]-rad+1:c[0]+rad+1, c[1]-rad:c[1]+rad], #down
	# 						dists2[xRange, yRange,2] + dists2Tot[c[0]-rad:c[0]+rad, c[1]-rad+1:c[1]+rad+1], #right
	# 						dists2[xRange, yRange,3] + dists2Tot[c[0]-rad:c[0]+rad, c[1]-rad-1:c[1]+rad-1]], axis=0) #left
	# imshow(dists2Tot*(dists2Tot < 500)*mask)
	# imshow(dists2Tot)

	for i in range(100):
		# dists2Tot[1:obj2Size[0]-1, 1:obj2Size[1]-1] = np.min([dists2Tot[1:-1, 1:-1],
		# 					dists2[:,:,0]+dists2Tot[0:-2, 1:-1]+1, #up
		# 					dists2[:,:,1]+dists2Tot[2:,   1:-1]+1, #down
		# 					dists2[:,:,2]+dists2Tot[1:-1, 2:]+1, #right
		# 					dists2[:,:,3]+dists2Tot[1:-1, 0:-2]+1], axis=0) #left
		# dists2Tot[1:-1, 1:-1] = np.min([dists2Tot[1:-1, 1:-1],
		# 					dists2[:,:,0]+dists2Tot[0:-d-1, 1:-1]+1, #up
		# 					dists2[:,:,1]+dists2Tot[d+1:,   1:-1]+1, #down
		# 					dists2[:,:,2]+dists2Tot[1:-1, d+1:]+1, #right
		# 					dists2[:,:,3]+dists2Tot[1:-1, 0:-d-1]+1], axis=0) #left
		dists2Tot[1:-1, 1:-1] = np.min([dists2Tot[1:-1, 1:-1],
							dists2[:,:,0]+dists2Tot[0:-d-1, 1:-1]-1, #up
							dists2[:,:,1]+dists2Tot[d+1:,   1:-1]-1, #down
							dists2[:,:,2]+dists2Tot[1:-1, d+1:]-1, #right
							dists2[:,:,3]+dists2Tot[1:-1, 0:-d-1]-1], axis=0) #left	
		dists2Tot[c[0]-2:c[0]+1, c[1]-2:c[1]+1] = d2[c[0]-2:c[0]+1, c[1]-2:c[1]+1]
		# Extra +1 is needed to try to keep pixels stationary
	imshow(dists2Tot*(dists2Tot < 500)*mask)


	# t = dists2Tot*(dists2Tot < 500)*mask
	# figure(2)
	# h = np.histogram(t)
	# plot(h[1][1:], h[0])

	# Crawl distance image to find largest distance
	maxDist = dists2Tot[(dists2Tot < 500)*mask].max()


#----------------------------------------------------

if 0:
	#get widths
	objIndex = 0
	im = out1[objects1[objIndex][0]] == objects1[objIndex][1]
	inds = np.nonzero(im) # y,x
	mean = [inds[0].mean(), inds[1].mean()] # y,x
	diff_x = inds[1] - mean[1]
	diff_y = inds[0] - mean[0]
	plot(diff_x, -diff_y, '.')


	#--------------Plane Detection-------------------
	fig = figure(1)
	ax = fig.add_subplot(111)
	ax.imshow(mean2)

	points = []
	def onclick(event):
	    print 'x=%d, y=%d, xdata=%f, ydata=%f'%(
	        event.x, event.y, event.xdata, event.ydata)
	    points.append([event.ydata, event.xdata])
	    # return event.xdata, event.ydata

	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	while len(points) < 2:
	# 	print 'Click point on image'
		time.sleep(1)
	fig.canvas.mpl_disconnect(cid)
	points = np.array(points, dtype=int)
	xStart = points[0,0]
	yStart = points[0,1]
	inds = np.nonzero(mean2[points[0,0]:points[1,0], points[0,1]:points[1,1]])
	inds = [inds[0]+xStart, inds[1]+yStart]
	inds.append(mean2[inds])
	xyd = np.array(inds)
	xyz = depth2world(xyd[0],480-xyd[1],xyd[2])
	xyz = np.array([xyz[0], xyz[1], xyz[2]])
	Fp_trans = xyz.mean(1)
	xyz = xyz.T - Fp_trans

	u, s, v = np.linalg.svd(xyz, full_matrices=0)
	Fp_rot = np.array(v.T)
	up = Fp_rot[:,2]
	if up[1] < 0:
		Fp_rot = -1*Fp_rot
		up = -1*up

	peak_z = np.array([peak_z[1],peak_z[0],peak_z[2]])
	peak_z, Fp_rot, 
	pZp = F*peak_z
	pZp = Rp*peak_z + Fp
	np.array(np.asmatrix(Fp_rot.T)*np.asmatrix(peak_z).T)[:,0] + Fp_trans

	peak_z_new = np.asmatrix(Fp_rot)*np.asmatrix(peak_z - Fp_trans).T


	#--------------Pose Estimation-------------------

	# labelInds1, objects1
	# out2[objects2[1]]==labelInds2[1]
	inds = np.nonzero(out2[objects2[1]]==16)
	y = objects2[1][0].start
	x = objects2[1][1].start
	# xyd = [inds[0]+y, inds[1]+x]
	xyd = [inds[0]+y, inds[1]+x]
	xyd.append(d2[inds])
	xyd = np.array(xyd)
	peakIndD = np.argmin(xyd[0,:])
	peak_d = xyd[:,peakIndD]
	peak_z = depth2world(peak_d[1], 480-peak_d[0], peak_d[2])
	world2depth(peak_z)

	# xyd = np.array(xyd)
	# xyd = xyd[:, (np.nonzero(xyd[2,:] < 9999))[0]]
	# xyz = np.array(depth2world(xyd[1],xyd[0],xyd[2]))
	# xyz = np.array([x for x in xyz.T if x[2] < 9999.0]).T
	# peakInd = np.argmax(xyz[1,:])
	# peakXyz = xyz[:,peakInd]
	# peakXyd = world2depth(peakXyz[0],peakXyz[1],peakXyz[2])

	## Use EM to reposition body parts using this parts-based model.
	# Spring(s) should come from default position (and other joints?)
	# This convention continues to use z as depth
	# Might be easier to find the floor first and do measurements relative to that
	# Use head, chest, shoulders, arms, legs. (Tubes??)
	Skel = {'Head': [.3, 1.65, 0], 'Chest': [.3, 1, 0]}

	#-----------Top-down View------------------------
	# Too elaborate or unnecessary?

	## Steps:
	# 1) Extract location on wall (subset of image)
	#x2) Convert to x/y/z
	#x3) SVD, find minimum eig
	#x4) Repeat 1-3 for second face
	# 5) Find z-vector by cross(v1, v2)
	# 6) Create virtual camera above (looking at -z)
	# 7) Find transformation from camera -> virtual camera
	# 8) Convert points to new coordinates. (reverse lookup?) ( only do for moving things)

	# d1 = reader1.depthDataRaw
	# d2 = reader2.depthDataRaw




