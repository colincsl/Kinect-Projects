


import os, time, sys
import numpy as np
import cv, cv2
import scipy.ndimage as nd
import pdb
from math import floor
sys.path.append('/Users/colin/code/Kinect-Projects/icuRecorder/')
from icuReader import ICUReader as ICUReader

from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn import metrics

def constrain(img):
	min_ = np.min(img[np.nonzero(img)])
	max_ = img.max() #/2
	# max_ = 4000.0
	img = np.minimum(img, max_)				
	img[np.nonzero(img)] -= min_
	# print max_, min_, ((max_-min_)/256.0)
	img = np.array((img / ((max_-min_)/256.0)), dtype=np.uint8)
	# img = 256 - img
	img = np.array(img, dtype=np.uint8)
	return img

def extractPeople(img):
	x = cv2.medianBlur(img, 15)
	hist1 = np.histogram(x, 256)


	samples = np.random.choice(hist1[1][1:-1], 1000, p=hist1[0][1::]*1.0/np.sum(hist1[0][1::]))
	samples = np.sort(samples)
	tmp = np.array([samples, samples])
	D = distance.squareform(distance.pdist(tmp.T))
	S = 1 - (D / np.max(D))

	db = DBSCAN().fit(S, eps=0.95, min_samples=50)
	# core_samples = db.core_sample_indices_
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	# plot(samples)
	# plot(-30*labels+100)

	clusterLimits = []
	for i in range(-1, n_clusters_):
		min_ = np.min(samples[np.nonzero((labels==i)*samples)])
		max_ = np.max(samples[np.nonzero((labels==i)*samples)])
		clusterLimits.append([min_,max_])
	clusterLimits.sort()
	print clusterLimits

	d = np.zeros_like(img)
	tmp = np.zeros_like(img)
	labels = []
	for i in range(-1, n_clusters_):
		# tmp = np.nonzero((img > clusterLimits[i][0])*(img < clusterLimits[i][1]))
		# d[np.nonzero((img > clusterLimits[i][0])*(img < clusterLimits[i][1]))] = (i+2)*40
		tmp = (x > clusterLimits[i][0])*(x < clusterLimits[i][1])
		d[np.nonzero((x > clusterLimits[i][0])*(x < clusterLimits[i][1]))] = (i+2)
		# labels.append(nd.label(tmp))
		tmpLab = nd.label(tmp)
		if labels == []:
			labels = tmpLab
		else:
			labels = [labels[0]+((tmpLab[0]>0)*(tmpLab[0]+labels[1])), labels[1]+tmpLab[1]]

	objs = nd.find_objects(labels[0], labels[1])
	goodObjs = []
	for i in range(len(objs)):	
		if objs[i] != None:
			# px = nd.sum(d[objs[i]], labels[0][objs[i]], i)
			px = nd.sum(d, labels[0], i)
			# px = nd.sum(d>0, labels[0], i)
			# print "Pix:", px
			if px > 10000:
				goodObjs.append([objs[i], i])
				goodObjs.append([objs[i], i])

	d1A = d1.copy()
	# figure(1)
	for i in range(len(goodObjs)):
		# subplot(1, len(goodObjs)+1, i+1)
		# imshow(d[goodObjs[i][0]] == goodObjs[i][1])
		# imshow(d[goodObjs[i][0]] > 0)
		# imshow(labels[0] == goodObjs[i][1])
		d1A = np.maximum(d1A, (labels[0] == goodObjs[i][1])*8000)
		# d1A[goodObjs[i][0]] = np.maximum(d1A[goodObjs[i][0]], (d[goodObjs[i][0]]>0)*8000)
		# d1A[goodObjs[i]] = np.maximum(d1A[goodObjs[i]], (d[goodObjs[i]]==d[goodObjs[i]].max())*8000)
		# d1A[d[np.nonzero(d[goodObjs[i]])]] = 8000
		# title(i)

	# figure(2)	
	# subplot(1,2,1)
	# imshow(d)
	# subplot(1,2,2)
	# imshow(d1*(d1A==8000))

	return d1A



path = '/Users/colin/data/ICU_7March2012_Head/'
framerate = 20;
startTime = 1600;
reader1 = ICUReader(path, framerate, startTime, cameraNumber=0, viz=0, vizSkel=0)


path2 = '/Users/colin/data/ICU_7March2012_Foot/'
startTime2 = startTime+12;
reader2 = ICUReader(path2, framerate, startTime2, cameraNumber=1, viz=0, vizSkel=0)



depthImgs1 = []
depthImgs2 = []

for i in range(10):
	reader1.run()
	depthImgs1.append(reader1.depthDataRaw)
	reader2.run()
	depthImgs2.append(reader2.depthDataRaw)

depthImgs1 = np.dstack(depthImgs1)
depthImgs2 = np.dstack(depthImgs2)

mean1 = np.mean(depthImgs1, 2)
mean2 = np.mean(depthImgs2, 2)

framerate = 100;
startTime = 0;
reader1 = ICUReader(path, framerate, startTime, cameraNumber=0, viz=0, vizSkel=0)
startTime2 = startTime+12;
reader2 = ICUReader(path2, framerate, startTime2, cameraNumber=1, viz=0, vizSkel=0)

cv.NamedWindow("a")
cv.NamedWindow("b")
for i in range(300):
	reader1.run()
	reader2.run()
	d1 = reader1.depthDataRaw
	d2 = reader2.depthDataRaw

	diff1 = d1*(np.abs(d1 - mean1) > 500)
	diff2 = d2*(np.abs(d2 - mean2) > 500)
	diffDraw1 = constrain(diff1)
	diffDraw2 = constrain(diff2)

	out1 = extractPeople(diffDraw1)
	dTMP = constrain(d1*(out1==8000))
	cv2.imshow("a", dTMP)

	out2 = extractPeople(diffDraw2)
	dTMP2 = constrain(d2*(out2==8000))
	cv2.imshow("b", dTMP2)
	cv2.waitKey(33)		





