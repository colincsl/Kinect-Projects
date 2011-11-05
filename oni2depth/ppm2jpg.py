

import cv

for i in xrange(10000):

	filename = 'depth_%04d'%i
	im = cv.LoadImage(filename+".ppm")
	
	#cv.Threshold(im, im, 150, 255, cv.CV_THRESH_TRUNC)
	
	cv.SaveImage("jpg/"+filename+".jpg", im)
	
	print "Img#: " + str(i)