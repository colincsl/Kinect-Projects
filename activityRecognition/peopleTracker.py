
import numpy as np, cv, cv2
import time, os
from copy import deepcopy
import icuReader
from SkelPlay import *
from random import randint

import pdb


class Tracker:
	def __init__(self, name="PeopleTracker", dir_="."):
		self.people = []
		self.people_open = []
		self.name = name #+ "_" + str(int(time.time()))
		self.dir = dir_

	def run(self, coms_, slices, time, depthFilename, touches):
		# import copy
		coms = deepcopy(list(coms_))
		comLabels = []
		deleteTimeThresh = 5
		movingCount = 3

		if len(self.people_open) > 0:
			distMat = np.zeros([len(coms), len(self.people_open)])
			for j in xrange(len(coms)):
				for pp in xrange(len(self.people_open)):
					p = self.people_open[pp]
					#x,y,z dist
					distMat[j,pp] = np.sqrt(np.sum((np.array(coms[j])-np.array(p['moving_com'][-1]))**2))
					# distMat[j,pp] = np.sqrt(np.sum((np.array(coms[j])-np.array(p['com'][-1]))**2))

			distMatSort = np.argsort(distMat, axis=1)
			
			deleteLst = []		
			for i in xrange(len(coms)):
				prevInd = distMatSort[i,0]
				# if prevInd == i and distMat[i, prevInd] < 400:
				if distMat[i, prevInd] < 500:
					self.people_open[prevInd]['com'].append(coms[i])
					moving_com = np.mean(np.vstack([self.people_open[prevInd]['moving_com'][-movingCount:],coms[i]]), axis=0)
					self.people_open[prevInd]['moving_com'].append(moving_com) #moving avg
					self.people_open[prevInd]['slice'].append(slices[i])
					self.people_open[prevInd]['time'].append(time)
					self.people_open[prevInd]['filename'].append(depthFilename)
					touches = [y for x, y in zip(touches, range(len(touches))) if i in x]
					if len(touches) > 0:
						if 'touches' not in self.people_open[prevInd]:
							self.people_open[prevInd]['touches'] = []
						self.people_open[prevInd]['touches'].append([touches, len(self.people_open[prevInd]['filename'])])
					distMat[:,prevInd] = 9999
					distMatSort = np.argsort(distMat, axis=1)
					deleteLst.append(i)
					comLabels.append(self.people_open[prevInd]['label'])
			deleteLst.sort(reverse=True)
			for i in deleteLst:
						coms.pop(i)

		for j in xrange(len(coms)):
			print "New person"
			i=0
			while(i in comLabels):
				i += 1
			comLabels.append(i)			

			self.people_open.append({'com':[coms[j]], 'moving_com':[coms[j]], 'slice':[slices[j]], 
				'time':[time], 'filename':[depthFilename], 'label':i})

		# Convert old people to 'overall' instead of current
		deleteLst = []
		for j in xrange(len(self.people_open)):
			p = self.people_open[j]
			if p['time'][-1] < time-deleteTimeThresh:
				pAvg = np.array(np.mean(p['com'], 0)) #centroid over time
				timeDiff = p['time'][-1]-p['time'][0]
				self.people.append({"data":p, "start":p['time'][0], 
								"elapsed":timeDiff,
								"com":pAvg, 'label':p['label']})
				deleteLst.append(j)		
		deleteLst.sort(reverse=True) # remove last nodes first
		for j in deleteLst:
			self.people_open.pop(j)	

		return comLabels

	def finalize(self):
		deleteLst = []
		for j in range(len(self.people_open)):
			p = self.people_open[j]
			pAvg = np.array(np.mean(p['com'], 0))
			timeDiff = p['time'][-1]-p['time'][0]
			self.people.append({"data":p, "start":p['time'][0], 
							"elapsed":timeDiff,
							"com":pAvg, 'label':p['label']})
			deleteLst.append(j)
		deleteLst.sort(reverse=True) # remove last nodes first
		for j in deleteLst:
			self.people_open.pop(j)

		startTime = 99999
		endTime = 0
		for p in self.people:
			if p['start'] < startTime:
				startTime = p['start']
			if p['start']+p['elapsed'] > endTime:
				endTime = p['start']+p['elapsed']

		meta = {'start':startTime, 'end':endTime, 'elapsed':endTime-startTime,
				'count':len(self.people)}
		os.chdir(self.dir)
		print "Tracker saved to ", (self.dir+self.name)
		# pdb.set_trace()
		np.savez(self.name, data=self.people, meta=meta)


#------------------------------------------------------------------------

def showLabeledImage(data, ind_j, dir_, rgb=0):
	cv.NamedWindow("a")
	if rgb:
		file_ = data['data']['filename'][ind_j]
		img = icuReader.getRGBImage(dir_+file_[:file_.find(".depth")]+".rgb")
		# imgD=img[:,::-1, :]
		imgD=img
	else:
		img = icuReader.getDepthImage(dir_+data['data']['filename'][ind_j])
		imgD = icuReader.constrain(img, 500, 5000)
		imgD = np.dstack([imgD, imgD, imgD])

	# com_xyz =  np.array(data['data']['com'])
	com_xyz =  np.array(data['data']['moving_com'])
	com = list(world2depth(com_xyz).T) #COM in image coords
	# Add previous location markers
	r = 5
	for jj in range(ind_j):
		s = (slice(com[jj][0]-r, com[jj][0]+r),slice(com[jj][1]-r, com[jj][1]+r))
		imgD[s[0], s[1], 0] = 100
		imgD[s[0], s[1], 1:3] = 0
	# Add current location marker
	r = 10
	s = (slice(com[ind_j][0]-r, com[ind_j][0]+r),slice(com[ind_j][1]-r, com[ind_j][1]+r))
	colorInd = 2#randint(0, 2)
	if colorInd==0: # Vary the colors every new person
		r=255; g=0; b=0
	elif colorInd==1:
		r=0; g=255; b=0
	elif colorInd==2:
		r=0; g=0; b=255				
	imgD[s[0], s[1], 0] = r
	imgD[s[0], s[1], 1] = g
	imgD[s[0], s[1], 2] = b	
	# Add touch boxes
	touchEnabled = 1
	if touchEnabled:
		rad = 13
		cv2.putText(imgD, "Touches", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), thickness=1)
		for i in range(2):
			cv2.rectangle(imgD, (60-rad, 40*i+60-rad), (60+rad, 40*i+60+rad), [255, 255, 255])
	if 'touches' in data['data'].keys():
		rad = 10
		times = [x[1] for x in data['data']['touches']]
		if any(np.equal(times,ind_j)):
			for i in data['data']['touches'][np.argwhere(np.equal(times,ind_j))[0]][0]:
				cv2.circle(imgD, (60, 40*i+60), rad, [0, 255, 0], thickness=4)
	# Print time on screen
	time_ = data['data']['time'][ind_j]
	hours = int(time_ / 3600)
	minutes = int(time_ / 60) - hours*60
	seconds = int(time_) - 60*minutes - 3600*hours
	text = str(hours) + "hr " + str(minutes) + "min " + str(seconds) + "s"
	cv2.putText(imgD, text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
	# Print duration to screen
	text = "Dur: " + str(data['elapsed'])
	cv2.putText(imgD, text, (450, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
	#Show image
	cv2.imshow("a", imgD)
	cv2.waitKey(1)


def labelData(filename, dir_='/Users/colin/data/ICU_7March2012_Head/', speed=3):
	playData(filename, dir_, speed, label=True)

def playData(filename, dir_='/Users/colin/data/ICU_7March2012_Head/', speed=3, label=False, filterbox=None):

	data_raw = np.load(filename)
	p = data_raw['data']
	m = data_raw['meta']
	name = filename[:filename.find(".")] + "_labeled"
	p = filterEvents(p)

	cv.NamedWindow("a")
	imgD = np.zeros([480,640, 3]) # Dislay each image.

	for ii in range(len(p)): # For each segmented person
		data = p[ii] # person data
		# if data['elapsed'] > 10: # filter really short segments
		if len(data['data']['time']) > 5:
			com = np.array([data['com']])
			# com_uv = list(world2depth(com).T)[0] #COM in image coords
			play = 1
			for j in xrange(0, len(data['data']['time']), speed):

				showLabeledImage(data, j, dir_)

				# Get label
				if label and (play or (j > len(data['data']['time'])-speed)):
					lab = raw_input("Label (Play: p; Prev Err: pe [label]; Save: s): ")
					if lab == "p":
						play = 0
					elif lab[:2] == "pe":
						errorFrames.append(ii)
						data['label'] = lab[3:]
					elif lab[0] == "s":
						np.savez(name, data=p, meta=m)
						play = 0
					else:
						data['label'] = lab
						break

				time.sleep(.01)
			# imgD[:,:,:] = 255
			# cv2.imshow("a", imgD)
			# ret = cv2.waitKey(1)
			# time.sleep(.05)

			if label:
				np.savez(name, data=p, meta=m)


def filterEvents(data, filterbox=[]):
	data = [x for x in data if len(x['data']['time']) > 20] #5

	# # Stuff on this side of the bed gets errased
	if 0:
		filterbox = [[0,240],[300, 680],[0, 4000]] # x, y, z
		com = np.array([x['com'] for x in data])
		com = list(world2depth(com).T) #COM in image 
		data = [data[y] for x,y in zip(com, range(len(data))) if not (filterbox[0][0] < x[0] < filterbox[0][1] and filterbox[1][0] < x[1] < filterbox[1][1] and filterbox[2][0] < x[2] < filterbox[2][1])]

	return data

#------------------------------------------------------------------------


if 0:
	labels = {1:'group', 2:'talking', 3:'observing', 4:'read', 5:'procedure', 6:'unrelated'}

	os.chdir('/Users/colin/code/Kinect-Projects/activityRecognition/')

	labelData("1.npz", '/Users/colin/data/ICU_7March2012_Head/', speed=20)
	playData("1.npz", '/Users/colin/data/ICU_7March2012_Head/', speed=10)
	playData("1.npz", '/Volumes/ICU/ICU_7March2012_Head/', speed=10)
	labelData("1.npz", '/Volumes/ICU/ICU_7March2012_Head/', speed=20)
	

	labelData("2_800s.npz", '/Users/colin/data/ICU_7March2012_Foot/')
	playData("2.npz", '/Users/colin/data/ICU_7March2012_Foot/')
	playData("2.npz", '/Volumes/ICU/ICU_7March2012_Foot/', speed=10)
	labelData("2.npz", '/Volumes/ICU/ICU_7March2012_Foot/', speed=20)


if 0:
	if 0:
		data = np.load('1_labeled.npz')
		data = np.load('1.npz')
		dir_ = '/Users/colin/data/ICU_7March2012_Head/'
	else:
		data = np.load('2_800s_labeled_good.npz')
		data = np.load('2.npz')
		dir_ = '/Users/colin/data/ICU_7March2012_Foot/'

	pOrig = data['data']
	p = filterEvents(pOrig)
	m = data['meta']	

	totalEventCount = len(p) # includes errors / really short events

	# labels = set()
	labels = np.array([(x['label']) for x in p if 'label' in x.keys()], dtype=str)
	labelNames = unique(labels)

	# Get counts for each label
	counts = [[], []]
	for i in xrange(len(labelNames)):
		counts[0].append(np.sum(np.repeat(labelNames[i], len(labels)) == labels))
		counts[1].append(labelNames[i])

	# Get indices for each label
	inds = []
	for i in xrange(len(labelNames)):
		inds.append([y for x,y in zip(p, range(0,len(p))) if 'label' in x.keys() and x['label']==labelNames[i]])

	# Show label images
	for lab in xrange(len(labelNames)):
		figure(lab)
		for i in xrange(len(inds[lab])):
			subplot(2,int(len(inds[lab])/2),i)	
			img = icuReader.getDepthImage(dir_+p[inds[lab][i]]['data']['filename'][0])
			imshow(img)
			axis('off')
			title(labelNames[lab])

	### Time ###
	# Get time for each label
	labeledTimes = []
	for i in xrange(len(labelNames)):
		labeledTimes.append(np.sum([p[x]['elapsed'] for x in inds[i]]))

	# Time histogram
	times = [x['elapsed'] for x in p if x['elapsed'] > 10]
	times_hist = np.histogram(times, bins=20, range=[10,150])
	plot(times_hist[0])
	xticks(range(0,len(times_hist[1]), 2), times_hist[1][::2]); xlabel('Time (s)');

	# Average times
	eventCount = np.sum([1 for x in p if x['elapsed'] > 10])	
	totalEventTime = np.sum([x['elapsed'] for x in p])
	totalValidEventTime = np.sum([x['elapsed'] for x in p if x['elapsed'] > 10])
	totalTime = p[-1]['start']+p[-1]['elapsed']
	avgEventTime = totalValidEventTime / eventCount
	avgTime = totalTime / eventCount

	# Play segment
	tmp = [y for x,y in zip(p, range(len(p))) if x['elapsed'] > 600] # time outliers
	i = tmp[1]
	# img = icuReader.getDepthImage(dir_+p[i]['data']['filename'][0])
	# imshow(img)
	speed = 2
	for j in xrange(0, len(p[i]['data']['filename']), speed):
		showLabeledImage(p[i], j, dir_)


if 0:

	# Create time-space
	# timeEvents = {}
	# for i in range(len(p)):
	# 	datum = p[i]
	# 	for j in datum['data']['time']:
	# 		if j not in timeEvents.keys():
	# 			timeEvents[j] = [i]
	# 		else:
	# 			if i not in timeEvents[j]:
	# 				timeEvents[j].append(i)

	# Create time-space
	timeEvents = {}
	for i in xrange(len(p)):
		datum = p[i]
		for j in xrange(len(datum['data']['time'])):
			t = datum['data']['time'][j]
			if t not in timeEvents.keys():
				timeEvents[t] = {i:[j]} #Event, event-time
			else:
				if i not in timeEvents[t].keys():
					timeEvents[t][i] = [j]
				else:
					timeEvents[t][i].append(j)


	max_ = 0
	figure(2)
	for i in timeEvents.keys():
		if len(timeEvents[i]) > max_:
			max_ = len(timeEvents[i])
			argmax_ = timeEvents[i][0]
			argmaxs_ = timeEvents[i]
		bar(int(i/60), len(timeEvents[i]))
	totalTime = p[-1]['start']+p[-1]['elapsed']
	axis([0, int(totalTime/60), 0, max_+1])
	title('Person count at each timestep', fontsize=20)
	xlabel('Time (min)', fontsize=18)
	ylabel('# People', fontsize=18)
	xticks(fontsize=16)
	yticks(fontsize=16)

	figure(3)
	ids = []
	for i in range(len(p)):
		start = p[i]['data']['time'][0]
		end = p[i]['data']['time'][-1]
		l = p[i]['label']
		plot([start, end], [l,l], linewidth=20)

	# Show images
	for i in range(len(argmaxs_)):
		showLabeledImage(p[argmaxs_[i]], 0, dir_)
	# Show each target in a frame
	k=0
	for i in range(len(timeEvents[k])):
		for j in range(len(p[timeEvents[k][i]]['data']['time'])):
			showLabeledImage(p[timeEvents[k][i]], j, dir_)
			time.sleep(.5)
	# Show video over time
	for i in xrange(100):
		if i in timeEvents:
			showLabeledImage(p[timeEvents[i][0]], 0, dir_)
			time.sleep(.1)

	# Show overhead view
	figure(2)
	# for i in range(len(argmaxs_)):
		# com = p[argmaxs_[i]]['com']
	colors = 'rgbkcy'
	for i in xrange(30000):
		if i in timeEvents:
			for j in xrange(len(timeEvents[i])):
				com = p[timeEvents[i][j]]['com']
				plot(-com[0], com[2], 'o', color=colors[j%6])
	title('Location of people over time (Camera 1)', fontsize=20)
	xlabel('X (mm)', fontsize=18)
	ylabel('Z (mm)', fontsize=18)
	xticks(fontsize=16)
	yticks(fontsize=16)
	axis('equal')


	frameCounts = [len(x['data']['time']) for x in p]

	## Find where labels triggered
	touches = [x for x,y in zip(p, range(len(p))) if 'touches' in x['data'].keys()]
	touchInds = [y for x,y in zip(p, range(len(p))) if 'touches' in x['data'].keys()]

	for i in range(1, len(touches)):
		for j in range(0, len(touches[i]['data']['time']), 3):
			showLabeledImage(touches[i], j, dir_)


	touchTmp1 = [y for x,y in zip(np.array(touches[0]['data']['touches'])[:,0], np.array(touches[0]['data']['touches'])[:,1]) if 0 in x]
	touchTmp2 = [y for x,y in zip(np.array(touches[0]['data']['touches'])[:,0], np.array(touches[0]['data']['touches'])[:,1]) if 1 in x]	

	touch0 = np.zeros(len(p))
	touch1 = np.zeros(len(p))
	for i in touchInds:
		touch0[i] = np.sum([1 for x in np.array(p[i]['data']['touches'])[:,0] if 0 in x])
		touch1[i] = np.sum([1 for x in np.array(p[i]['data']['touches'])[:,0] if 1 in x])

	## Find arclengths
	arclengths = np.empty([len(p)])
	for i in xrange(len(p)):
		sum_ = 0
		for j in xrange(1, len(p[i]['data']['com'])):
			sum_ += np.sqrt(np.sum((p[i]['data']['com'][j]-p[i]['data']['com'][j-1])**2))
		arclengths[i] = sum_
	times = np.array([x['elapsed'] for x in p])
	lengthTime = arclengths / times
	subplot(3,1,1); plot(arclengths); title('Arclengths')
	subplot(3,1,2); plot(times); title('Times')
	subplot(3,1,3); plot(lengthTime); title('Length/Times')

	coms = np.array([x['com'] for x in p])
	center = np.array([250, -200, 1000])
	comsRad = np.sqrt(np.sum((coms - center)**2, 1))

	# -------------------Features-------------------------------------------------
	# do personCount
	arcMax = np.max(arclengths)
	lengthTimeMax = np.max(lengthTime)
	touch0Max = np.max(touch0)
	touch1Max = np.max(touch1)
	frameCountMax = np.max(frameCounts)
	comsMax = np.max(coms_rad)
	# features = np.array([arclengths, lengthTime, touch0, touch1]).T
	featuresNorm = np.array([arclengths/arcMax, lengthTime/lengthTimeMax, \
							touch0/touch0Max, touch1/touch1Max, \
							frameCount/frameCountMax, comsRad/comsMax]).T
	# featuresNorm = np.array([arclengths/arcMax, touch0/touch0Max, touch1/touch1Max]).T

	# --------------------------------------------------------------------------------
	## Clustering

	X = featuresNorm
	COLORS = 'rgbkcyr'
	eventLabels = [int(x['label']) for x in p]
	labelColorsTmp = [COLORS[int(x['label'])] for x in p]

	 scatter(X_iso[:,0], X_iso[:,1])
	from sklearn import manifold
	X_iso = manifold.Isomap(10, 2).fit_transform(X)
	X_iso = manifold.Isomap(10, 2).fit_transform(X)
	figure(1); scatter(X_iso[:,0], X_iso[:,1], c=labelColorsTmp); title('Isomap') 
	X_lle = manifold.LocallyLinearEmbedding(10, 2).fit_transform(X)
	figure(2); scatter(X_lle[:,0], X_lle[:,1], c=labelColorsTmp); title('LLE')

	# LLE doesn't seperate as well

	from scipy.spatial import distance
	from sklearn.cluster import DBSCAN
	from sklearn.cluster import AffinityPropagation as AP
	from sklearn.cluster import MeanShift as MS
	from sklearn import metrics

	# Xl = X
	Xl = X_iso
	# Xl = X_lle
	D = distance.squareform(distance.pdist(Xl))
	S = 1 - (D / np.max(D))
	# clust = DBSCAN().fit(S, eps=0.85, min_samples=5)
	# clust = AP().fit(S)
	clust = MS().fit(S)

	labels = clust.labels_
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	print n_clusters, "cluster"

	# Print labeled manifold
	figure(1);
	for i in range(1, n_clusters):
		plot(Xl[labels==i,0], Xl[labels==i,1], 'o')
		# scatter(Xl[labels==i,0], Xl[labels==i,1], c=labelColorsTmp[labels==i])

	labelCounts = []
	labelInds = []
	for i in range(n_clusters):
		labelCounts.append(np.sum(labels == i))
		labelInds.append([y for x,y in zip(labels, range(len(labels))) if x == i])

	for i in labelInds[1]:
		for j in xrange(0,len(p[i]['data']['time']), 20):
			showLabeledImage(p[i], j, dir_)
			
	#  ---------------------------------------------------------


	#  ---------------------------------------------------------
	## SVM
	from sklearn import svm as SVM

	Y = np.zeros(len(p))
	for i in touchInds:
		Y[i] = 1
	Ystart = deepcopy(Y)

	svm = SVM.NuSVC(nu=.2, probability=True)
	# svm = SVM.NuSVC(nu=.5, kernel='poly')

	for i in xrange(10):
		svm.fit(X, Y)
		Y = svm.predict(X)

	probs = svm.predict_proba(X)

	changed = [y for x, y in zip(Y!=Ystart, range(len(Y))) if x]
	changedToPos = [x for x in changed if Y[x]]
	changedToNeg = [x for x in changed if not Y[x]]

	for i in changedToNeg:
		for j in xrange(0,len(p[i]['data']['time']), 20):
			showLabeledImage(p[i], j, dir_)
		print i

	for i in range(len(p)):
		if not Y[i]:
			for j in xrange(0,len(p[i]['data']['time']), 20):
				showLabeledImage(p[i], j, dir_)
			print i


	#  ---------------------------------------------------------
	timeEvents = {}
	for i in xrange(len(p)):
		datum = p[i]
		for j in xrange(len(datum['data']['time'])):
			t = datum['data']['time'][j]
			if t not in timeEvents.keys():
				timeEvents[t] = {i:[j]} #Event, event-time
			else:
				if i not in timeEvents[t].keys():
					timeEvents[t][i] = [j]
				else:
					timeEvents[t][i].append(j)
	# Fill out rest of times
	maxTimeData = p[timeEvents[timeEvents.keys()[-1]].keys()[0]]
	maxTime = maxTimeData['start']+maxTimeData['elapsed']
	for i in xrange(maxTime):
		if i not in timeEvents.keys():
			timeEvents[i] = {}

	#  --------------HMM--------------------------------------

labels = {1:'group', 2:'talking', 3:'observing', 4:'read', 5:'procedure', 6:'unrelated'}

prevStates = np.zeros(len(labels.keys()), dtype=bool)
newStates = np.zeros(len(labels.keys()), dtype=bool)
labelCounts = np.zeros([len(labels.keys()), 2,2], dtype=float)

# Generate A matrix
for secInd in xrange(1, maxTime):
	newStates[:] = False
	for eventInd in timeEvents[secInd]:
		datum = p[eventInd]
		label = int(datum['label'])
		newStates[label-1] = True

	for i in xrange(len(labels)):
		if prevStates[i] and newStates[i]:   # P(t|t)
			labelCounts[i,0,0] += 1
		elif ~prevStates[i] and newStates[i]:  # P(t|~t)
			labelCounts[i,0,1] += 1
		elif prevStates[i] and ~newStates[i]:  # P(~t|t)
			labelCounts[i,1,0] += 1
		elif ~prevStates[i] and ~newStates[i]: # P(~t|~t)
			labelCounts[i,1,1] += 1

	prevStates[:] = newStates[:]

AMats = labelCounts / labelCounts[0].sum()

# Generate pi matrix
piMats = np.empty([len(labels.keys())])
for i in xrange(len(labels.keys())):
	piMats[i] = Amats[i][0,:].sum() / Amats[i][1,:].sum()

## Generate B matrix

# Plot Distributions
for i in range(6):
	subplot(2,3,i)
	h = hist(featuresNorm[:,i])
	plot(h[1][1:], h[0])

# SVD of distributions
_,_,v = svd(featuresNorm, full_matrices=0)
basis = v[0]
pcaFeatures = np.dot(featuresNorm, v[0])
scatter(range(len(pcaFeatures)), pcaFeatures, c=labelColorsTmp)
scatter(np.array(range(len(pcaFeatures)))*0, pcaFeatures, c=labelColorsTmp)

from sklearn.mixture import GMM
labelMeans = np.empty([len(labels.keys())])
labelCovars = np.empty([len(labels.keys())])
labelDevs = np.empty([len(labels.keys())])
m1 = GMM(1)
for i in labels.keys():
	m1.fit(pcaFeatures[np.nonzero(np.equal(eventLabels, i))])
	labelMeans[i-1] = m1.means
	labelCovars[i-1] = m1.covars[0][0][0]


def gaussProb(data, mean, covar):
	return (1 / np.sqrt(2*np.pi*covar) * np.exp( -(data - mean)**2 / (2*covar))) / 2.124



#  --------------Test----------------
current = piMats
for i in xrange(maxTime):
	val = np.dot(featuresNorm[i,:], basis)
	BMat = gaussProb(1.0, labelMeans, labelCovars)

	current *= BMat


#### Current feature data is in event space ###
#!!!#### Create feature data in time space ######!!#





