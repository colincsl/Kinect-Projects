
import os, time, sys
import numpy as np
import cv, cv2

sys.path.append('/Users/colin/code/Kinect-Projects/activityRecognition/')
from icuReader import ICUReader
from peopleTracker import Tracker
from SkelPlay import *

import pdb


class Features:
	img = []
	labelImg = []
	personSlices = []
	labels = []
	allFeatureNames = ['basis', 'binary', 'viz']
	touchAreas = []
	segmentsTouched = []

	def __init__(self, featureList=[]):
		self.featureList = []
		self.addFeatures(featureList)

	def run(self, img, labelImg, personSlices, labels):
		self.img = img
		self.labelImg = labelImg
		self.personSlices = personSlices
		self.labels = labels
		self.segmentsTouched = []

		self.calculateFeatures()		
		return self.img, self.coms, self.bases, self.segmentsTouched


	def addFeatures(self, strList = []):
		for i in strList:
			if i in self.allFeatureNames:
				self.featureList.append(i)
			else:
				print i, " is not a valid feature name"

	def addTouchEvent(self, center, radius):
		if 'touch' not in self.featureList:
			self.featureList.append('touch')
		self.touchAreas.append([center, radius])

	# def addTouchBoxEvent(self, box):
	# 	self.featureList.append('touch')
	# 	self.touchAreas.append(box)		

	def calculateFeatures(self):
		if 'basis' in self.featureList:
			self.calculateBasis()
		if 'touch' in self.featureList:
			for i in self.touchAreas:
				self.touchEvent(i[0], i[1])
		if 'viz' in self.featureList:
			self.vizBasis()



	def calculateBasis(self):
		vecs_out = []
		com_out = [] #center of mass
		com_xyz_out = []
		bounds = []
		if len(self.personSlices) > 0:
			for objIndex in xrange(len(self.personSlices)):
				inds = np.nonzero(self.labelImg[self.personSlices[objIndex]] == self.labels[objIndex])
				offsetX = self.personSlices[objIndex][0].start #Top left corner of object slice
				offsetY = self.personSlices[objIndex][1].start
				inds2 = [inds[0]+offsetX, inds[1]+offsetY]
				depVals = self.img[inds2]
				inds2.append(depVals)
				inds2 = np.transpose(inds2)
				# following is unnecessary unless axis has already been painted on the image
				# inds2 = [x for x in inds2 if x[2] != 0]
				xyz = np.array(depth2world(np.array(inds2)))
				inds2 = np.transpose(inds2)

				# Get bounding box
				x = [np.min(xyz[:,0]), np.max(xyz[:,0])]
				y = [np.min(xyz[:,1]), np.max(xyz[:,1])]
				z = [np.min(xyz[:,2]), np.max(xyz[:,2])]
				# bounds.append([[x[0],y[0],z[0]], [x[1],y[1],z[1]]])
				bounds.append([[x[0],y[0],z[0]], \
							   [x[0],y[0],z[1]], \
							   [x[0],y[1],z[0]], \
							   [x[0],y[1],z[1]], \
							   [x[1],y[0],z[0]], \
							   [x[1],y[0],z[1]], \
							   [x[1],y[1],z[0]], \
							   [x[1],y[1],z[1]]])

				com = xyz.mean(0)
				xyz = xyz - com

				u, s, v = np.linalg.svd(xyz, full_matrices=0)
				v = v.T
				if v[1,0] < 0:
					v = -1*v
				vecs = []
				for i in xrange(3):
					vecs.append(v[:,i])
				vecs_out.append(vecs)

				com_xyz_out.append(com)
				com_uv = world2depth(np.array([com]))
				com_uv = com_uv.T[0]
				com_out.append(com_uv)

		self.bases = vecs_out
		self.coms = com_out
		self.coms_xyz = com_xyz_out
		self.bounds = bounds


	def vizBasis(self):
		# Display axis on the original image
		for ind in range(len(self.coms)):
			com = self.coms_xyz[ind]
			v = np.array(self.bases[ind])
			spineStart = [com[0], com[1], com[2]]
			spineLine = []
			for axis in xrange(3):
				spine = v[:,axis]
				for i in xrange(-100*0, 100, 2):
					spineLine.append(spineStart+spine*i)
			spineLine = np.array(spineLine)
			spineLine_xyd = np.array(world2depth(spineLine))

			if 0:
				#Show 3d structure
				fig = figure(2)
				ax = fig.add_subplot(111, projection='3d')
				# ax.cla()
				ax.scatter(xyz[0,::4], xyz[1,::4], xyz[2,::4])
				ax.scatter(spineLine[:,0], spineLine[:,1], spineLine[:,2], 'g')
				xlabel('x')
				ylabel('y')
			
			if len(spineLine_xyd) > 0:
				# Make sure all points are within bounds
				inds = [x for x in range(len(spineLine_xyd[0])) if (spineLine_xyd[0, x] >= 0 and spineLine_xyd[0, x] < 480 and spineLine_xyd[1, x] >= 0 and spineLine_xyd[1, x] < 640)]
				spineLine_xyd = spineLine_xyd[:,inds]
				self.img[spineLine_xyd[0], spineLine_xyd[1]] = 0#spineLine_xyd[2]


	def touchEventBox(self, box=[]):
		# check if the person and event boxes intersect
		# box format: [[x1,y1,z1],[x2,y2,z1]] (two corners)
		segmentsTouched = []
		for ind in range(len(self.coms)):
			c = self.bounds[ind]
			# x
			# pdb.set_trace()
			if c[0][0] < box[0][0] < c[1][0] or c[0][0] < box[1][0] < c[1][0] \
				or box[0][0] < c[0][0] < box[1][0] or box[0][0] < c[1][0] < box[1][0]:

				# y
				if c[0][1] < box[0][1] < c[1][1] or c[0][1] < box[1][1] < c[1][1] \
					or box[0][1] < c[0][1] < box[1][1] or box[0][1] < c[1][1] < box[1][1]:
					# z
					if c[0][2] < box[0][2] < c[1][2] or c[0][2] < box[1][2] < c[1][2] \
						or box[0][2] < c[0][2] < box[1][2] or box[0][2] < c[1][2] < box[1][2]:
						segmentsTouched.append(ind)

		self.segmentsTouched.append(segmentsTouched)


	def touchEvent(self, center=[332, -104, 983], radius=50):
		# check if the person and event circle (w/in radius) overlap
		# pdb.set_trace()
		sqr_dist = np.sqrt(np.sum((self.bounds - np.array(center))**2, 2))
		touched = np.nonzero(np.any(sqr_dist < radius, 1))[0]

		self.segmentsTouched.append(touched)





	def play(self):
		# Look at histograms of centered data projected on each basis
		dists = np.sqrt(np.sum(xyz**2, 1))
		dists = np.sqrt((xyz**2))
		dx[inds2[0], inds2[1]] = dists[:, 0]
		# Project to each principal axis
		axis = 1
		d = np.asarray(np.asmatrix(dists) * np.asmatrix(v[:,axis]).T)
		dx = np.array(dx, dtype=np.int16)
		dx[inds2[0], inds2[1]] = -1*d[:,0]#-d[:, 0]
		figure(1)
		imshow(dx)
		h = np.histogram(d, 100, [-500, 500])
		figure(2)
		plot(h[1][1::],h[0])
		figure(3)
		plot(d)


