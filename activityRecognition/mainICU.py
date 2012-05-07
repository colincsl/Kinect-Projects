
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

# import pyximport
# pyximport.install()
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
path = '/Users/colin/data/ICU_7March2012_Head/'
framerate = 20;
startTime = 1600;
path2 = '/Users/colin/data/ICU_7March2012_Foot/'
#---------------------------------------------
startTime2 = startTime+12;
reader1 = ICUReader(path, framerate, startTime, cameraNumber=0, viz=0, vizSkel=0, skelsEnabled=0)
reader2 = ICUReader(path2, framerate, startTime2, cameraNumber=1, viz=0, vizSkel=0, skelsEnabled=0)

depthImgs1 = []
depthImgs2 = []
depthStackInd1 = 0
depthStackInd2 = 0
for i in xrange(10):
	reader1.run()
	depthImgs1.append(reader1.depthDataRaw)
	reader2.run()
	depthImgs2.append(reader2.depthDataRaw)

depthImgs1 = np.dstack(depthImgs1)
depthImgs2 = np.dstack(depthImgs2)

mean1 = getMeanImage(depthImgs1)
mean2 = getMeanImage(depthImgs2)
m1 = constrain(mean1, 500, 4000)
m2 = constrain(mean2, 500, 4500)

#-----------------------------------------
#--------------MAIN-----------------------
#-----------------------------------------

framerate = 30
startTime = 6900#2000 #2x procedure: 6900, 7200
# startTime = 1350#2000
startTime2 = startTime+12
serial = True
if 0: #Serial
	reader1 = ICUReader(path, framerate, startTime, cameraNumber=0, viz=0, vizSkel=0, skelsEnabled=0, serial=1)
	reader2 = ICUReader(path2, framerate, startTime2, cameraNumber=1, viz=0, vizSkel=0, skelsEnabled=0, serial=1)
else: #Real-time
	reader1 = ICUReader(path, framerate, startTime, cameraNumber=0, viz=0, vizSkel=0, skelsEnabled=0, serial=0)
	reader2 = ICUReader(path2, framerate, startTime2, cameraNumber=1, viz=0, vizSkel=0, skelsEnabled=0, serial=0)

vizWin = 1
if vizWin:
	cv.NamedWindow("a")
	# cv.NamedWindow("a_seg")
	cv.NamedWindow("b")
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
featureExt1.addTouchEvent([250, -200, 1000], 350)
featureExt1.addTouchEvent([-175, -150, 1150], 300)
# featureExt1.addTouchEvent([0, 0, 2100], 200)
featureExt2 = Features(['basis', 'viz']) #feature extractor

# for i in xrange(1):
# while(1):
histVecs = []
start = time.time()
# while (len(reader1.allPaths) > 0):
while(1):
# if 1:

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
			diffDraw1 = d1c*(diff > 10)			
			out1, objects1, labelInds1 = extractPeople_2(diffDraw1)
			if len(labelInds1) > 0:
				d1, com1, vecs1, touched1 = featureExt1.run(d1, out1, objects1, labelInds1)
				ornCompare = orientationComparison(vecs1)
				com1_xyz = featureExt1.coms_xyz
				t = reader1.timeMin*60 + reader1.timeSec
				ids = tracker1.run(com1_xyz, objects1, t, reader1.depthFilename, touched1, vecs1, ornCompare)

				# for i in xrange(len(ids)):
				# 	# pdb.set_trace()
				# 	v = ids[i]
				# 	if len(histVecs) <= v:
				# 		histVecs.append([])
				# 	histVecs[v].append(vecs1[i])

				# print "People: ", ids
			extremaInds = []
			for i in xrange(len(labelInds1)):
				try:
					extrema = getExtrema(objects1, labelInds1, out1, d1, com1, featureExt1, i)
					extremaInds.append(extrema)
				except:
					print "Error getting extrema"

			tEnd = time.time()
			# print "Time 1: ", tEnd - tStart

			if vizWin:
				out1b = np.zeros_like(out1, dtype=np.uint8)+255
				out1b = np.dstack([out1b, out1b, out1b])
				d1c = constrain(d1, 500, 4000)
				d1c = np.dstack([d1c, d1c, d1c])
				# Draw binary sensors
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


				for exSet in extremaInds:
					for ex in exSet:
						d1c[ex[0]-2:ex[0]+2, ex[1]-2:ex[1]+2] = 255

				cv2.imshow("a", d1c)
				out1 *= 10 * (out1>0)
				out1s = np.array(np.dstack([out1*10, out1*11, out1*12]), dtype=uint8)
				# cv2.imshow("a_seg", out1b)
		except:
			print 'Error in camera 1'

	if 0:
		try:
		# if 1:
			tStart = time.time()
			reader2.run()
			d2 = reader2.depthDataRaw
			d2c = constrain(d2, 500, 4500)
			diffDraw2 = d2c*((m2 - d2c) > 50)*((m2 - d2c) < 225)
			out2, objects2, labelInds2 = extractPeople_2(diffDraw2)
			if len(objects2) > 0:
				d2, com2, vecs2 = peopleBasis(d2, out2, objects2, labelInds2, viz=0)
				com2_xyz = depth2world(np.array(com2))
				t = reader2.timeMin*60 + reader2.timeSec
				ids2 = tracker2.run(com2_xyz, objects2, t, reader2.depthFilename)
			tEnd = time.time()
			print "Time 2: ", tEnd - tStart	
		
			if vizWin:
				out2b = np.zeros_like(out2, dtype=np.uint8)+255
				out2b = np.dstack([out2b, out2b, out2b])
				d2c = np.dstack([d2c, d2c, d2c])
				for i in xrange(len(objects2)):
					# out1[out1==labelInds1[i]] = (ids[i]+1)*50
					if 1:
						d2c[out2==labelInds2[i], ids2[i]%3] = (ids2[i]+1)*50
						d2c[out2==labelInds2[i], (ids2[i]+1)%3] = 0
						d2c[out2==labelInds2[i], (ids2[i]+2)%3] = 0
					else:
						d2c[out2==labelInds2[i], 1] = 100
						d2c[out2==labelInds2[i], 2] = 255
						d2c[out2==labelInds2[i], 0] = 0					
					# out1b[out1==labelInds1[i], ids[i]%3] = (ids[i]+1)*50
				# cv2.imshow("a", constrain(d1, 500, 4000))
				cv2.imshow("b", d2c)
				out2 *= 10 * (out2>0)
				out2s = np.array(np.dstack([out2*10, out2*11, out2*12]), dtype=uint8)
				cv2.imshow("b_seg", out2b)

		except:
			print 'Error in camera 2'

	ret = cv2.waitKey(1)
	if ret > 0:
		break

	print "Time: ", (reader1.currentDirTime - reader1.startDirTime) , ", left: " , len(reader1.allPaths)

tracker1.finalize()
end = time.time()
print end - start
# tracker2.finalize()





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
np.savez('tmpPerson.npz', {'objects':objects1, 'labels':labelInds1, 'out':out1, 'd':d1, 'com':com1, 'features':featureExt1})


if 0:
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

	# import pyximport
	# pyximport.install()
	import dijkstrasGraph

	saved = np.load('tmpPerson.npz')['arr_0'].tolist()
	objects1 = saved['objects']; labelInds1=saved['labels']; out1=saved['out']; d1=saved['d']; com1=saved['com'];featureExt1=saved['features']

	objects = objects1
	labelInds = labelInds1
	out = out1
	d=d1
	com = com1
	extrema = []
	ind = 1
	mask = out[objects[ind]]==labelInds[ind]
	mask_erode = nd.binary_erosion(out[objects[ind]]==labelInds[ind], iterations=5)
	# imshow(out2[objects2[ind]]==labelInds2[ind])
	objTmp = np.array(d[objects[ind]])#, dtype=np.uint16)
	# objTmp *= mask

	# cv.PyrDown(cv.fromarray(objTmp), cv.fromarray(objTmpLow))
	# objTmp = objTmpLow

	obj2Size = np.shape(objTmp)
	x = objects[ind][0].start # down
	y = objects[ind][1].start # right
	c = np.array([com[ind][0] - x, com[ind][1] - y])
	current = [c[0], c[1]]
	xyz = featureExt1.xyz[ind]

	for i in xrange(5):
	# if 1:
		com_xyz = depth2world(np.array([[current[0]+x, current[1]+y, d1[current[0]+x, current[1]+y]]]))[0]
		dists = np.sqrt(np.sum((xyz-com_xyz)**2, 1))
		inds = featureExt1.indices[ind]
		
		distsMat = np.zeros([obj2Size[0],obj2Size[1]], dtype=uint16)		
		distsMat = ((-mask)*499)
		distsMat[inds[0,:]-x, inds[1,:]-y] = dists 		
		objTmp = distsMat

		dists2 = np.empty([obj2Size[0]-2,obj2Size[1]-2,4], dtype=int16)
		dists2[:,:,0] = objTmp[1:-1, 1:-1] - objTmp[0:-2, 1:-1]#up
		dists2[:,:,1] = objTmp[1:-1, 1:-1] - objTmp[2:, 1:-1]#down
		dists2[:,:,2] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 2:]#right
		dists2[:,:,3] = objTmp[1:-1, 1:-1] - objTmp[1:-1, 0:-2]#left
		dists2 = np.abs(dists2)


		dists2Tot = np.zeros([obj2Size[0],obj2Size[1]], dtype=int16)+9999		
		maxDists = np.max(dists2, 2)
		distThresh = 30
		outline = np.nonzero(maxDists>distThresh)
		mask[outline[0]+1, outline[1]+1] = 0

		dists2Tot[dists2Tot > 0] = 9999
		dists2Tot[-mask] = 15000
		dists2Tot[current[0], current[1]] = 0

		visitMat = np.zeros_like(dists2Tot, dtype=uint8)
		visitMat[-mask] = 255

		t = time.time()
		dists2Tot = dijkstrasGraph.dijkstras(dists2Tot, visitMat, dists2, current)
		print time.time()-t

		dists2Tot *= mask_erode
			# print current
		# imshow(dists2Tot*(dists2Tot < 800))
		maxInd = np.argmax(dists2Tot*(dists2Tot<500))
		maxInd = np.unravel_index(maxInd, dists2Tot.shape)

		extrema.append([maxInd[0], maxInd[1]])
		current = [maxInd[0], maxInd[1]]

	for i in extrema:
		dists2Tot[i[0]-3:i[0]+3, i[1]-3:i[1]+3] = 799
	imshow(dists2Tot*(dists2Tot < 800))




	
	distsDone = copy.deepcopy(dists2Tot)
	distsDone = distsDone + (-mask)*9999
	current = [c[0], c[1]]
	test = [140, 60]
	current = test
	path = []
	path.append([current[0], current[1]])


	done = 1000
	while(done > 0):		
		done -= 1

		# if np.min(distsDone[current[0]-1:current[0]+2, current[1]-1:current[1]+2]+np.array([[1,0,1],[0,1,0],[1,0,1]])*9999) >= distsDone[current[0], current[1]]:
		if 1:
			distsDone[current[0], current[1]] = -999
			direc = np.argmax(distsDone[current[0]-1:current[0]+2, current[1]-1:current[1]+2]+np.array([[1,0,1],[0,1,0],[1,0,1]])*(-9999))

			print direc
			if direc == 1:
				current[0] -= 1
			elif direc == 7:
				current[0] += 1
			elif direc == 5:
				current[1] += 1
			elif direc == 3:
				current[1] -= 1	

			if current == path[-1]:
				break
			print current, direc

			path.append([current[0], current[1]])

		else:
			break
		

	imshow(distsDone*(distsDone < 1000))
	imshow(dists2Tot*(dists2Tot < 1000))
	distsDone = copy.deepcopy(dists2Tot)
	distsDone = distsDone + (-mask)*9999
	current = [c[0], c[1]]
	path = []
	path.append(current)
	conn = np.argmin(dists2, 2)





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




