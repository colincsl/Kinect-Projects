
import os, time, sys
import numpy as np
import cv, cv2
import scipy.ndimage as nd
import pdb
from math import floor
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn import metrics
sys.path.append('/Users/colin/code/Kinect-Projects/activityRecognition/')
from icuReader import ICUReader as ICUReader
from peopleTracker import Tracker
from SkelPlay import *

#---------------------------------------------
path = '/Users/colin/data/ICU_7March2012_Head/'
framerate = 20;
startTime = 1600;
path2 = '/Users/colin/data/ICU_7March2012_Foot/'
#---------------------------------------------

def constrain(img, mini=-1, maxi=-1): #500, 4000
	if mini == -1:
		min_ = np.min(img[np.nonzero(img)])
	else:
		min_ = mini
	if maxi == -1:
		max_ = img.max()
	else:
		max_ = maxi

	img = np.clip(img, min_, max_)
	img -= min_
	if max_ == 0:
		max_ = 1
	img = np.array((img * (255.0 / (max_-min_))), dtype=np.uint8)

	return img

def extractPeople(img):
	if len(img) == 1:
		img = img[0]
	img = img*nd.binary_opening(img>0, iterations=5)
	img = cv2.medianBlur(img, 5)
	# hist1 = np.histogram(img, 128)
	hist1 = np.histogram(img, 64)
	if (np.sum(hist1[0][1::]) > 100):
		samples = np.random.choice(np.array(hist1[1][1:-1], dtype=int), 1000, p=hist1[0][1::]*1.0/np.sum(hist1[0][1::]))
		samples = np.sort(samples)
	else:
		return np.zeros_like(img), []

	tmp = np.array([samples, np.zeros_like(samples)])
	D = distance.squareform(distance.pdist(tmp.T, 'cityblock'))
	# D = distance.squareform(distance.pdist(tmp.T, 'chebyshev'))
	S = 1 - (D / np.max(D))

	db = DBSCAN().fit(S, eps=0.95, min_samples=50)
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	clusterLimits = []
	# for i in range(-1, n_clusters_):
	max_ = 0
	for i in xrange(1, n_clusters_):		
		min_ = np.min(samples[np.nonzero((labels==i)*samples)])
		max_ = np.max(samples[np.nonzero((labels==i)*samples)])
		clusterLimits.append([min_,max_])
	if max_ != 255:
		clusterLimits.append([max_, 255])
	clusterLimits.append([0, min(clusterLimits)[0]])
	clusterLimits.sort()

	d = np.zeros_like(img)
	tmp = np.zeros_like(img)
	labels = []
	for i in xrange(0, n_clusters_):
		tmp = (img > clusterLimits[i][0])*(img < clusterLimits[i][1])
		d[np.nonzero((img > clusterLimits[i][0])*(img < clusterLimits[i][1]))] = (i+2)
		tmpLab = nd.label(tmp)
		if labels == []:
			labels = tmpLab
		else:
			labels = [labels[0]+((tmpLab[0]>0)*(tmpLab[0]+tmpLab[1])), labels[1]+tmpLab[1]]

	objs = nd.find_objects(labels[0])
	goodObjs = []
	for i in xrange(len(objs)):	
		if objs[i] != None:
			# px = nd.sum(d, labels[0], i)
			px = nd.sum(labels[0][objs[i]]>0)
			# print "Pix:", px
			if px > 7000:
				goodObjs.append([objs[i], i+1])

	d1A = np.zeros_like(img)
	for i in xrange(len(goodObjs)):
		## Plot all detected people
		# subplot(1, len(goodObjs)+1, i+1)
		# imshow(d[goodObjs[i][0]] == goodObjs[i][1])
		# imshow(d[goodObjs[i][0]] > 0)
		# imshow(labels[0] == goodObjs[i][1])

		d1A = np.maximum(d1A, (labels[0] == goodObjs[i][1])*(i+1))		
		# d1A[goodObjs[i][0]] = np.maximum(d1A[goodObjs[i][0]], (d[goodObjs[i][0]]>0)*8000)

	return d1A, goodObjs


def extractPeople_2(im):
	im_ = im[1:480, 1:640]
	im = np.array(im, dtype=float)
	grad_x = im[1:480, 1:640] - im[1:480, 0:639]
	grad_y = im[1:480, 1:640] - im[0:479, 1:640]
	grad_g = np.maximum(np.abs(grad_y), np.abs(grad_x))
	grad_bin = (grad_g < 20)*(im_ > 0)

	for i in xrange(4):
		grad_bin = nd.binary_erosion(grad_bin)

	labels = nd.label(grad_bin)
	objs = nd.find_objects(labels[0])
	# get rid of noise (if count is too low)
	objs = [x for x in zip(objs, (range(1, len(objs)+1))) if nd.sum(grad_bin[x[0]]) > 5000]
	if len(objs) > 0:
		objs, goodLabels = zip(*objs) # unzip objects
	else:
		goodLabels = []

	return labels[0], objs, goodLabels

def peopleBasis(depthImg, labelImg, objects, viz=0):
	# Find principal components of person
	img = depthImg
	vecs_out = []
	com_out = []
	if len(objects) > 0:
		for objIndex in xrange(len(objects)):
			# inds = np.nonzero(labelImg[objects1[objIndex][0]])
			inds = np.nonzero(labelImg[objects[objIndex]])
			inds2 = [inds[0], inds[1]]
			depVals = depthImg[inds2]
			inds2.append(depVals)
			xyz = np.array(depth2world(inds2))
			xyz = xyz[:, np.nonzero(xyz[0,:])].reshape(3,-1)
			xyz = np.array([xyz[1], xyz[0], xyz[2]])
			xyz = xyz.T - xyz.mean(1)

			u, s, v = np.linalg.svd(xyz, full_matrices=0)
			v = v.T
			vecs = []
			for i in xrange(3):
				vecs.append(v[:,i])
			vecs_out.append(vecs)

			com = nd.center_of_mass(labelImg[objects[objIndex]])
			com = [int(com[0] + objects[objIndex][0].start),
					int(com[1] + objects[objIndex][1].start)]
			# com.append(depthImg[com[0], com[1]])
			com.append(np.mean(depthImg[objects[objIndex]]))
			com_out.append(com)
			if viz:
				com_xyz = depth2world(com)
				spineStart = [com_xyz[0], com_xyz[1], float(com_xyz[2])]
				# spineStart = [xyz[0].mean(), xyz[1].mean(), xyz[2].mean()]
				# print com
				spineLine = []
				for axis in xrange(3):
					spine = v[:,axis]
					for i in xrange(-100, 100, 3):
						spineLine.append(spineStart+spine*i)
				spineLine = np.array(spineLine)

				if 0:
					fig = figure(2)
					ax = fig.add_subplot(111, projection='3d')
					# ax.cla()
					ax.scatter(xyz[0,::4], xyz[1,::4], xyz[2,::4])
					ax.scatter(spineLine[:,0], spineLine[:,1], spineLine[:,2], 'g')
					xlabel('x')
					ylabel('y')

				spineLine_xyd = np.array(world2depth(spineLine))
				if len(spineLine_xyd) > 0:
					inds = [x for x in range(len(spineLine_xyd[0])) if (spineLine_xyd[0, x] >= 0 and spineLine_xyd[0, x] < 480 and spineLine_xyd[1, x] >= 0 and spineLine_xyd[1, x] < 640)]
					# pdb.set_trace()
					spineLine_xyd = spineLine_xyd[:,inds]
					# spineLine_xyd = [x for x in spineLine_xyd if (x[0] >= 0 and x[0] < 480 and x[1] >= 0 and x[1] < 640)]
					# spineLine_xyd = np.array(spineLine_xyd)
					img[spineLine_xyd[0], spineLine_xyd[1]] = 10000#spineLine_xyd[2]

	return img, com_out, vecs_out

def getMeanImage(depthImgs):
	mean_ = np.mean(depthImgs, 2)
	mean_ = mean_*(~nd.binary_dilation(mean_==0, iterations=3))

	#Close holes in images
	# inds = nd.distance_transform_edt(mean1<2000, return_distances=False, return_indices=True)
	inds = nd.distance_transform_edt(mean_<500, return_distances=False, return_indices=True)
	i2 = np.nonzero(mean_<500)
	i3 = inds[:, i2[0], i2[1]]
	mean_[i2] = mean_[i3[0], i3[1]] # For all errors, set to avg 

	return mean_



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#----------------Init---------------------

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
m2 = constrain(mean2, 500, 6000)


#-----------------------------------------
framerate = 30
startTime = 0#2000
startTime2 = startTime+12
reader1 = ICUReader(path, framerate, startTime, cameraNumber=0, viz=0, vizSkel=0, skelsEnabled=0)
reader2 = ICUReader(path2, framerate, startTime2, cameraNumber=1, viz=0, vizSkel=0, skelsEnabled=0)

vizWin = 1
if vizWin:
	cv.NamedWindow("a")
	cv.NamedWindow("a_seg")
	cv.NamedWindow("b")
	cv.NamedWindow("b_seg")

# from multiprocessing import Pool, Queue, Process
### Can't pass numpy arrays into processes! Must use ctypes
## See numpy-sharedmem

dir_ = '/Users/colin/code/Kinect-Projects/activityRecognition/'
# dir_ = os.getcwd()
tracker1 = Tracker('1', dir_)
tracker2 = Tracker('2', dir_)

# while(1):
for i in xrange(100):
	# tStart = time.time()
	# tEnd = time.time()
	# print "Time 1: ", tEnd - tStart

	# try:
	tStart = time.time()
	reader1.run()	
	d1 = reader1.depthDataRaw
	d1c = constrain(d1, 500, 4000)
	diffDraw1 = d1c*(np.abs(m1 - d1c) > 50)*((m1 - d1c) < 225)	
	out1, objects1, labelInds1 = extractPeople_2(diffDraw1)
	d1, com1, vecs1 = peopleBasis(d1, out1, objects1, viz=1)
	t = reader1.timeMin*60 + reader1.timeSec
	tracker1.run(com1, objects1, t, reader1.depthFilename)
	tEnd = time.time()
	print "Time 1: ", tEnd - tStart	
	# except:
	# 	print 'Error in camera 1'

	# imshow(d1[people1_open[3][1]['slice']])
	try:
		tStart = time.time()
		reader2.run()
		d2 = reader2.depthDataRaw
		d2c = constrain(d2, 500, 6000)
		diffDraw2 = d2c*((m2 - d2c) > 50)*((m2 - d2c) < 225)
		out2, objects2, labelInds2 = extractPeople_2(diffDraw2)
		d2, com2, vecs2 = peopleBasis(d2, out2, objects2, viz=0)
		t = reader2.timeMin*60 + reader2.timeSec
		tracker2.run(com2, objects2, t, reader2.depthFilename)
		tEnd = time.time()
		print "Time 2: ", tEnd - tStart	
	except:
		print 'Error in camera 1'

	# dSeg = constrain(out1, out1.min(), out1.max())
	# dSeg2 = constrain(out2, out2.min(), out2.max())

	if vizWin:
		cv2.imshow("a", constrain(d1, 500, 4000))
		cv2.imshow("b", constrain(d2, 500, 6000))
		out1 = out1 * np.floor(255/out1.max())
		out2 = out2 * np.floor(255/out2.max())
		cv2.imshow("a_seg", out1)
		cv2.imshow("b_seg", out2)
		cv2.waitKey(1)

tracker1.finalize()
tracker2.finalize()
#Finally, add everyone in open list to 'overall' list
# deleteLst = []
# for j in xrange(len(people1_open)):
# 	p = people1_open[j]
# 	pAvg = np.array([0,0,0])
# 	for pp in p:
# 		pAvg += np.array(pp['com'])
# 	pAvg /= len(p)
# 	timeDiff = p[-1]['time']-p[0]['time']
# 	people1.append({"data":p, "start":p[0]['time'], 
# 					"elapsed":timeDiff,
# 					"com":pAvg})
# 	deleteLst.append(j)
# deleteLst.sort(reverse=True) # remove last nodes first
# for j in deleteLst:
# 	people1_open.pop(j)

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

ind = 1
mask = out2[objects2[ind]]==labelInds2[ind]
mask = out2[objects2[ind]]==24
objTmp = np.array(d2[objects2[ind]])#, dtype=np.uint16)
# objTmpLow = np.empty([objTmp.shape[0]/2, objTmp.shape[1]/2], dtype=uint16)
# cv.PyrDown(cv.fromarray(objTmp), cv.fromarray(objTmpLow))
# objTmp = objTmpLow
# dists2 = np.zeros_like(d2[objects2[0]])
obj2Size = np.shape(objTmp)
x = objects2[ind][0].start # down
y = objects2[ind][1].start # right
c = np.array([com2[ind][0] - x, com2[ind][1] - y])

# Floyd-Warshall Algorithm
# d = 3; dH = 2; dI = 1
d = 3; dH = 2
dists2 = np.empty([obj2Size[0]-(d+1),obj2Size[1]-(d+1),4], dtype=int16)
dists2Tot = np.zeros([obj2Size[0],obj2Size[1]], dtype=int16)+9999
# dists2Tot = objTmp*mask + (mask==0)*9999
# dists2Tot = 9999

# dists2Tot[mask] = 999
# objTmp[c[0]-1, c[1]-1] = 0
# dists2[:,:,0] = objTmp[1:obj2Size[0]-1, 1:obj2Size[1]-1] - objTmp[0:obj2Size[0]-2, 1:obj2Size[1]-1]#up
# dists2[:,:,1] = objTmp[1:obj2Size[0]-1, 1:obj2Size[1]-1] - objTmp[2:obj2Size[0], 1:obj2Size[1]-1]#down
# dists2[:,:,2] = objTmp[1:obj2Size[0]-1, 1:obj2Size[1]-1] - objTmp[1:obj2Size[0]-1, 2:obj2Size[1]]#right
# dists2[:,:,3] = objTmp[1:obj2Size[0]-1, 1:obj2Size[1]-1] - objTmp[1:obj2Size[0]-1, 0:obj2Size[1]-2]#left

dists2[:,:,0] = objTmp[dH:obj2Size[0]-dH, dH:obj2Size[1]-dH] - objTmp[0:obj2Size[0]-(d+1), dH:obj2Size[1]-dH]#up
dists2[:,:,1] = objTmp[dH:obj2Size[0]-dH, dH:obj2Size[1]-dH] - objTmp[(d+1):obj2Size[0], dH:obj2Size[1]-dH]#down
dists2[:,:,2] = objTmp[dH:obj2Size[0]-dH, dH:obj2Size[1]-dH] - objTmp[dH:obj2Size[0]-dH, (d+1):obj2Size[1]]#right
dists2[:,:,3] = objTmp[dH:obj2Size[0]-dH, dH:obj2Size[1]-dH] - objTmp[dH:obj2Size[0]-dH, 0:obj2Size[1]-(d+1)]#left

# dists2[c[0]-1, c[1]-1] = 0
dists2 = np.abs(dists2)
dists2Min = np.min(np.abs(dists2), 2)
# dists2Tot[c[0]-1:c[0]+1, c[1]-1:c[1]+1] = dists2Min[c[0]-1:c[0]+1, c[1]-1:c[1]+1]
dists2Tot = objTmp
dists2Tot[c[0]-2:c[0]+1, c[1]-2:c[1]+1] = objTmp[c[0]-2:c[0]+1, c[1]-2:c[1]+1]
dists2Tot[c[0], c[1]] = 0



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
	# dists2Tot[dH:-dH, dH:-dH] = np.min([dists2Tot[dH:-dH, dH:-dH],
	# 					dists2[:,:,0]+dists2Tot[0:-d-1, dH:-dH]+1, #up
	# 					dists2[:,:,1]+dists2Tot[d+1:,   dH:-dH]+1, #down
	# 					dists2[:,:,2]+dists2Tot[dH:-dH, d+1:]+1, #right
	# 					dists2[:,:,3]+dists2Tot[dH:-dH, 0:-d-1]+1], axis=0) #left
	dists2Tot[dH:-dH, dH:-dH] = np.min([dists2Tot[dH:-dH, dH:-dH],
						dists2[:,:,0]+dists2Tot[0:-d-1, dH:-dH]-1, #up
						dists2[:,:,1]+dists2Tot[d+1:,   dH:-dH]-1, #down
						dists2[:,:,2]+dists2Tot[dH:-dH, d+1:]-1, #right
						dists2[:,:,3]+dists2Tot[dH:-dH, 0:-d-1]-1], axis=0) #left	
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




