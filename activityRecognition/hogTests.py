import sys
sys.path.append("/Users/colin/libs/pyvision/build/lib.macosx-10.7-intel-2.7/")
from vision import features
from scikits.learn import svm
import scipy.ndimage as nd

from scipy.io.matlab import loadmat

#-------
''' Make picture of positive HOG weights.'''
def HOGpicture(w, bs, positive=True):
	# w=feature, bs=size, 

	# construct a "glyph" for each orientaion
	bim1 = np.zeros([bs,bs])
	bim1[:,round(bs/2):round(bs/2)+1] = 1

	bim = np.zeros([bim1.shape[0],bim1.shape[1], 9])
	bim[:,:,1] = bim1
	for i in range(2,10):
		bim[:,:,i-1] = nd.rotate(bim1, -(i-1)*20, reshape=False) #crop?

	# make pictures of positive weights bs adding up weighted glyphs
	shape_ = w.shape
	if positive:
		w[w<0] = 0
	else:
		w[w>0] = 0
	# im = np.zeros([bs*shape_[0], bs*shape_[1]])
	im = np.zeros([bs*shape_[1], bs*shape_[0]])
	for i in range(1,shape_[0]):
		for j in range(1,shape_[1]):
			for k in range(9):
				# im[(i-1)*bs:i*bs, (j-1)*bs:j*bs] += bim[:,:,k]*w[i,j,k]
				im[(j-1)*bs:j*bs,(i-1)*bs:i*bs] += bim[:,:,k]*w[i,j,k]


	return im


from scikits.learn import svm

# Get training data
trainingFeatures = io.loadmat('bodyPart_HOGFeatures.mat')
bodyLabels = [x for x in trainingFeatures if x[0]!='_']

allFeatures = []
allFeaturesLab = []
for lab in bodyLabels:
	for feat in trainingFeatures[lab]:
		allFeatures.append(feat.reshape([-1]))
		allFeaturesLab.append(lab)
featureCount = len(allFeaturesLab)

'''Train'''
svms = []
for feat,i in zip(allFeatures, range(len(allFeaturesLab))):
	labels = np.zeros(len(allFeaturesLab))
	labels[i] = 1
	# svm_ = svm.NuSVC(nu=.2, probability=True)
	svm_ = svm.SVC(probability=True)
	svm_.fit(allFeatures, labels)
	svms.append(deepcopy(svm_))
'''Test'''
svmPredict = np.empty([featureCount,featureCount])
for feat,i in zip(allFeatures, range(len(allFeaturesLab))):
	for j in xrange(featureCount):
		svmPredict[i,j] = svms[j].predict_proba(feat)[0][0]



	labels = np.zeros(len(allFeaturesLab))
	labels[i] = 1
	# svm_ = svm.NuSVC(nu=.2, probability=True)
	svm_ = svm.SVC(probability=True)
	svm_.fit(allFeatures, labels)










for k in range(9):
	print k
	tmpBoxRot = allBoxesRot[k]
	# tmpBoxRot = np.nan_to_num(tmpBoxRot)
	tmpBoxRot[tmpBoxRot>0] -= tmpBoxRot[tmpBoxRot>0].min()
	tmpBoxRot[tmpBoxRot>0] /= tmpBoxRot.max()
	# tmpBoxRot[tmpBoxRot>0] = np.log(tmpBoxRot[tmpBoxRot>0])
	tmpBoxD = np.dstack([tmpBoxRot, tmpBoxRot, tmpBoxRot])
	f = features.hog(tmpBoxD, 4)

	# figure(1); subplot(3,3,k+1)
	figure(1); imshow(tmpBoxRot)

	# figure(2); subplot(3,3,k+1)
	# imshow(f[:,:,i-1])

	im = HOGpicture(f, 4)
	figure(2); imshow(im, interpolation='nearest')


