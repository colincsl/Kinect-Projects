import numpy as np
sys.path.append('/Users/colin/libs/visionTools/slic-python/')
import slic

def MinimumSpanningTree(distMatrix, startNode=0):
	#Input 2D NxN numpy array
	nodeList = [startNode]
	edgeList = []
	edgeListDict = {}
	nodes = range(distMatrix.shape[0])
	nodes.remove(startNode)
	currentNode = startNode

	edgeListDict[0] = []
	for i in nodes:
		edgeListDict[i] = []

	while nodes != []:
		minDist = np.inf
		closestNode = -1
		currentNode = -1
		for curr in nodeList:
		# curr = currentNode
		# if 1:
			for n in nodes:
				if distMatrix[n,curr] < minDist:
					minDist = distMatrix[n,curr]
					currentNode = curr
					closestNode = n
		if closestNode >= 0:
			nodes.remove(closestNode)
			nodeList.append(closestNode)
			edgeList.append([currentNode, closestNode])
			edgeListDict[currentNode].append(closestNode)
			edgeListDict[closestNode].append(currentNode)

			# currentNode = closestNode		
			# print minDist
		else:
			print 'Error'
			break	
	return edgeList, edgeListDict



def PruneEdges(edgeDict, maxLength = 2):
	nodeList = edgeDict.keys()
	deletedInds = []
	for n in nodeList:
		if len(edgeDict[n]) == 1:
			length = 1
			subtree = [n]
			n2 = edgeDict[n][0]

			while len(edgeDict[n2]) == 2 and length < maxLength:
				subtree.append(n2)
				# n2 = edgeDict[n2][0]
				n2 = [x for x in edgeDict[n2] if x not in subtree][0]
				print n, n2, subtree
				length += 1
			subtree.append(n2)

			# if len(edgeDict[n2]) > 2:
			if len(subtree) > 1:
				for i in range(0, len(subtree)-1):
					print i, subtree
					edgeDict[subtree[i]].remove(subtree[i+1])
					edgeDict[subtree[i+1]].remove(subtree[i])
					deletedInds.append(subtree[i])

	return edgeDict, deletedInds


def GetLeafLengths(edgeDict):
	nodeList = edgeDict.keys()
	trees = []
	for n in nodeList:
		if len(edgeDict[n]) == 1:
			subtree = [n]
			n2 = edgeDict[n][0]
			# print n2, len(edgeDict[n2])
			while len(edgeDict[n2]) == 2:
				subtree.append(n2)
				nodeList = [x for x in edgeDict[n2] if x not in subtree]
				if len(nodeList) > 0:
					n2 = nodeList[0]
				else:
					break

			subtree.append(n2)
			if len(subtree) >= 1:
				trees.append(subtree)
	return trees


# # Test on ICU data
# image_argb = dstack([d1c, d1c, d1c, d1c])
# image_argb = dstack([m1, m1, m1, m1])
# # region_labels = slic.slic_s(image_argb, 10000, 1)

# image_argb = dstack([diffDraw1,diffDraw1,diffDraw1,diffDraw1])
# region_labels = slic.slic_n(image_argb, 100, 0)
# slic.contours(image_argb, region_labels, 1)
# plt.imshow(image_argb[:, :, 0])


# regions = slic.slic_n(np.array(np.dstack([im,im[:,:,2]]), dtype=uint8), 50,10)

dists2Tot[dists2Tot>1000] = 1000

im8bit = (d[objects[ind]]*mask_erode)
im8bit = im8bit / np.ceil(im8bit.max()/256.0)

im4d = np.dstack([mask_erode, im8bit, im8bit, im8bit])
# im4d = np.dstack([mask_erode, dists2Tot, dists2Tot, dists2Tot])
# im4d = np.dstack([mask_erode, im8bit, dists2Tot, mask_erode])
regions = slic.slic_n(np.array(im4d, dtype=uint8), 100,5)
# regions = slic.slic_s(np.array(im4d, dtype=uint8), 300,2)
regions *= mask_erode
imshow(regions)

avgColor = np.zeros([regions.shape[0],regions.shape[1],3])
# avgColor = np.zeros([regions.shape[0],regions.shape[1],4])

regionCount = regions.max()
regionLabels = []
for i in range(regionCount):
	if 1: #if using x/y/z
		meanPos = posMat[regions==i,:].mean(0)
	if 0: # If using distance map
		meanPos = np.array([posMat[regions==i,:].mean(0)[0],
							posMat[regions==i,:].mean(0)[1],
							# posMat[regions==i,:].mean(0)[2],
							(dists2Tot[regions==i].mean())])		
	if 0: # If using depth only
		meanPos = np.array([(np.nonzero(regions==i)[0].mean()),
					(np.nonzero(regions==i)[1].mean()),
					(im8bit[regions==i].mean())])
	avgColor[regions==i,:] = meanPos
	if not np.isnan(meanPos[0]) and meanPos[0] != 0.0:
		tmp = np.nonzero(regions==i)
		argPos = [int(tmp[0].mean()),int(tmp[1].mean())]
		regionLabels.append([i, meanPos, argPos])

#Reindex
regionCount = len(regionLabels)
for lab, j in zip(regionLabels, range(regionCount)):
	lab.append(j)
	# mapRegionToIndex.append(lab[0])

# (Euclidan) Distance matrix
distMatrix = np.zeros([regionCount, regionCount])
for i_data,i_lab in zip(regionLabels, range(regionCount)):
	for j_data,j_lab in zip(regionLabels, range(regionCount)):
		if i_lab <= j_lab:
			# distMatrix[i_lab,j_lab] = np.sqrt(((i_data[1][0]-j_data[1][0])**2)+((i_data[1][1]-j_data[1][1])**2)+.5*((i_data[1][2]-j_data[1][2])**2))
			distMatrix[i_lab,j_lab] = np.sqrt(np.sum((i_data[1]-j_data[1])**2))
distMatrix = np.maximum(distMatrix, distMatrix.T)
distMatrix += 1000*eye(regionCount)
# distMatrix[distMatrix > 400] = 1000
edges = distMatrix.argmin(0)

if 0:
	''' Draw edges based on closest node '''
	imLines = deepcopy(regions)
	for i in range(regionCount):
		pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
		cv2.circle(imLines, pt1, radius=0, color=125, thickness=3)

	for i in range(regionCount):
		pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
		pt2 = (regionLabels[edges[i]][2][1],regionLabels[edges[i]][2][0])
		cv2.line(imLines, pt1, pt2, 100)

mstEdges, edgeDict = MinimumSpanningTree(distMatrix)

# ''' Refine MST '''
# edgeDict, deletedInds = PruneEdges(edgeDict, maxLength=2)

# for i in deletedInds[-1::-1]:
# 	del regionLabels[i]

# #Reindex
# regionCount = len(regionLabels)
# for lab, j in zip(regionLabels, range(regionCount)):
# 	lab.append(j)
# 	# mapRegionToIndex.append(lab[0])

# # (Euclidan) Distance matrix
# distMatrix = np.zeros([regionCount, regionCount])
# for i_data,i_lab in zip(regionLabels, range(regionCount)):
# 	for j_data,j_lab in zip(regionLabels, range(regionCount)):
# 		if i_lab <= j_lab:
# 			# distMatrix[i_lab,j_lab] = np.sqrt(((i_data[1][0]-j_data[1][0])**2)+((i_data[1][1]-j_data[1][1])**2)+.5*((i_data[1][2]-j_data[1][2])**2))
# 			distMatrix[i_lab,j_lab] = np.sqrt(np.sum((i_data[1]-j_data[1])**2))
# distMatrix = np.maximum(distMatrix, distMatrix.T)
# distMatrix += 1000*eye(regionCount)
# edges = distMatrix.argmin(0)

# mstEdges, edgeDict = MinimumSpanningTree(distMatrix)



figure(1); imshow(objTmp[:,:,2])

''' Draw edges based on minimum spanning tree '''
imLines = deepcopy(regions)
for i in range(regionCount):
	pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
	cv2.circle(imLines, pt1, radius=0, color=125, thickness=3)

# Draw line for all edges
if 1:
	for i in range(len(mstEdges)):
		pt1 = (regionLabels[mstEdges[i][0]][2][1],regionLabels[mstEdges[i][0]][2][0])
		pt2 = (regionLabels[mstEdges[i][1]][2][1],regionLabels[mstEdges[i][1]][2][0])
		cv2.line(imLines, pt1, pt2, 100)
figure(2); imshow(imLines)

''' Draw line between all core nodes '''

# Draw circles
imLines = deepcopy(regions)
for i in range(regionCount):
	pt1 = (regionLabels[i][2][1],regionLabels[i][2][0])
	cv2.circle(imLines, pt1, radius=0, color=125, thickness=3)

leafPaths = GetLeafLengths(edgeDict)
leafLengths = [len(x) for x in leafPaths]
core = [x for x in edgeDict.keys() if len(edgeDict[x]) > 2]
branchesSet = set()
for i in leafPaths:
	for j in i:
		branchesSet.add(j)
core = np.sort(list(set(range(regionCount)).difference(branchesSet)))
# core = [x for x in edgeDict.keys() if len(edgeDict[x]) > 2]
for i in range(len(core)-1):
	pt1 = (regionLabels[core[i]][2][1], regionLabels[core[i]][2][0])
	pt2 = (regionLabels[core[i+1]][2][1],regionLabels[core[i+1]][2][0])
	cv2.line(imLines, pt1, pt2, 150)


# Draw line for all leafs
for i in range(len(leafPaths)):
	if len(leafPaths[i]) > 3:
		color = 125
	else:
		color = 100
	for j in range(len(leafPaths[i])-1):
		pt1 = (regionLabels[leafPaths[i][j]][2][1],regionLabels[leafPaths[i][j]][2][0])
		pt2 = (regionLabels[leafPaths[i][j+1]][2][1],regionLabels[leafPaths[i][j+1]][2][0])
		cv2.line(imLines, pt1, pt2, color)


#Draw head and hands
pt1 = (regionLabels[core[0]][2][1],regionLabels[core[0]][2][0])
cv2.circle(imLines, pt1, radius=10, color=150, thickness=1)

for i in xrange(len(leafLengths)):
	if leafLengths[i] >= 4:
		pt1 = (regionLabels[leafPaths[i][0]][2][1],regionLabels[leafPaths[i][0]][2][0])
		cv2.circle(imLines, pt1, radius=10, color=125, thickness=1)



figure(3); imshow(imLines)


