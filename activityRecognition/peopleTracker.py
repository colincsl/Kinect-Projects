
import numpy as np, cv, cv2
import time, os
import pdb
import copy
import icuReader


class Tracker:
	def __init__(self, name="PeopleTracker", dir_="."):
		self.people = []
		self.people_open = []
		self.name = name #+ "_" + str(int(time.time()))
		self.dir = dir_

	def run(self, coms_, slices, time, depthFilename):
		coms = copy.deepcopy(coms_)
		deleteTimeThresh = 10

		if len(self.people_open) > 0:
			distMat = np.zeros([len(coms), len(self.people_open)])
			for j in xrange(len(coms)):
				for pp in xrange(len(self.people_open)):
					p = self.people_open[pp]
					distMat[j,pp] = np.sqrt(np.sum((np.array(coms[j])-np.array(p['com'][-1]))**2))

			distMatSort = np.argsort(distMat, axis=1)
			
			deleteLst = []		
			for i in xrange(len(coms)):
				# pdb.set_trace()
				prevInd = distMatSort[i,0]
				# if prevInd == i and distMat[i, prevInd] < 400:
				if distMat[i, prevInd] < 300:
					self.people_open[prevInd]['com'].append(coms[i])
					self.people_open[prevInd]['slice'].append(slices[i])
					self.people_open[prevInd]['time'].append(time)
					self.people_open[prevInd]['filename'].append(depthFilename)
					distMat[:,prevInd] = 9999			
					distMatSort = np.argsort(distMat, axis=1)
					deleteLst.append(i)
			deleteLst.sort(reverse=True)
			for i in deleteLst:
						coms.pop(i)

		for j in xrange(len(coms)):
			print "New person"
			self.people_open.append({'com':[coms[j]], 'slice':[slices[j]], 
				'time':[time], 'filename':[depthFilename]})

		# Convert old people to 'overall' instead of current
		deleteLst = []
		for j in xrange(len(self.people_open)):
			p = self.people_open[j]
			if p['time'][-1] < time-deleteTimeThresh:
				pAvg = np.array(np.mean(p['com'], 0)) #centroid over time
				timeDiff = p['time'][-1]-p['time'][0]
				self.people.append({"data":p, "start":p['time'][0], 
								"elapsed":timeDiff,
								"com":pAvg})
				deleteLst.append(j)		
		deleteLst.sort(reverse=True) # remove last nodes first
		for j in deleteLst:
			self.people_open.pop(j)	

	def finalize(self):
		deleteLst = []
		for j in range(len(self.people_open)):
			p = self.people_open[j]
			pAvg = np.array(np.mean(p['com'], 0))
			timeDiff = p['time'][-1]-p['time'][0]
			self.people.append({"data":p, "start":p['time'][0], 
							"elapsed":timeDiff,
							"com":pAvg})
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



if 0:
	import numpy as np, cv, cv2
	import time, os
	import pdb
	import copy
	import icuReader

	# view
	# data = np.load('1.npz')
	data = np.load('2.npz')
	# p = data['people']
	p = data['data']
	m = data['meta']
	cv.NamedWindow("a")
	# cv.NamedWindow("b")
	# dir_ = '/Users/colin/data/ICU_7March2012_Head/'
	dir_ = '/Users/colin/data/ICU_7March2012_Foot/'
	imgD = np.zeros([1,1,1])
	for ii in range(len(p)):
		i = p[ii]
		com =  np.array(i['data']['com'])
		# plot(com[:,1],com[:,0], 'o')
		# axis([0, 640, 0, 480])
		if len(com) > 4:
			for j in xrange(len(com)):
				folder_ = i['data']['filename'][j][0:i['data']['filename'][j].find("_")]+"/"
				img = icuReader.getDepthImage(dir_+folder_+i['data']['filename'][j])
				imgD = icuReader.constrain(img, 500, 5000)
				imgD = np.dstack([imgD, imgD, imgD])
				r = 5
				for jj in range(j):
					# s = (slice(com[jj,0]-r, com[jj,0]+r),slice(com[jj,1]-r, com[jj,1]+r))
					s = (slice(com[jj,0]-r, com[jj,0]+r),slice(com[jj,1]-r, com[jj,1]+r))
					imgD[s[0], s[1], 0] = 100
					imgD[s[0], s[1], 1:3] = 0
				r = 10
				s = (slice(com[j,0]-r, com[j,0]+r),slice(com[j,1]-r, com[j,1]+r))
				if ii%3==0:
					r=255; g=0; b=0
				elif ii%3==1:
					r=0; g=255; b=0
				elif ii%3==2:
					r=0; g=0; b=255				
				imgD[s[0], s[1], 0] = r
				imgD[s[0], s[1], 1] = g
				imgD[s[0], s[1], 2] = b			
				cv2.imshow("a", imgD)
				ret = cv2.waitKey(1)
				time.sleep(.1)
			imgD[:,:,:] = 255
			cv2.imshow("a", imgD)
			ret = cv2.waitKey(1)
			time.sleep(.1)

