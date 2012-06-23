
import os
import cv, cv2
import numpy as np

### Rectification
## From: http://nicolas.burrus.name/index.php/Research/KinectCalibration
fx_d = 5.9421434211923247e+02
fy_d = 5.9104053696870778e+02
cx_d = 3.3930780975300314e+02
cy_d = 2.4273913761751615e+02
k1_d = -2.6386489753128833e-01
k2_d = 9.9966832163729757e-01
p1_d = -7.6275862143610667e-04
p2_d = 5.0350940090814270e-03
k3_d = -1.3053628089976321e+00

camMatrix = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0,0,1]])
distCoeffs = np.array([k1_d, k2_d, p1_d, p2_d, k3_d])
paths = ['/Users/colin/data/EdShoot/EdShoot1/jpgs_nn','/Users/colin/data/EdShoot/EdShoot2/jpgs_nn'] 


for k in paths:

	os.chdir(k)
	DirNames = os.listdir('.')
	DirNames = [x for x in DirNames if x[0]!= '.']

	for i in DirNames:
		os.chdir(i)
		fileNames = os.listdir('.')
		fileNames = [x for x in fileNames if x[0] != '.']
		
		for j in fileNames:
			im = cv2.imread(j)
			im2 = cv2.undistort(im, camMatrix, distCoeffs)
			name = "undist_" + j
			saveIm = cv.fromarray(im2)
			cv.SaveImage(name, saveIm)
		os.chdir('..')

