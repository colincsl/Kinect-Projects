
# Remove all files from path with .rgb extension

import os
import numpy as np


path = '/Users/colin/data/ICU_7March2012_Head/'
# playTypes = ['depth', 'rgb']
os.chdir(path)

dirNames = os.listdir('.')
dirNames = [x for x in dirNames if (x[0] != '.')] # make sure it's not a hidden file


# Get current directory
for i in range(len(dirNames)):
	os.chdir(path+dirNames[i])
	fileNames = os.listdir('.')
	
	if len(fileNames) > 3:
		periodInd = fileNames[0].find('.')
		fNamesBase = [x[0:periodInd] for x in fileNames]
		fNamesBase = np.unique(fNamesBase)
		fileNamesR = [x for x in fileNames if (x[-3:] == 'rgb') and x[0:periodInd] in fNamesBase] # make sure it's not a hidden file
		fileNamesD = [x for x in fileNames if (x[-5:] == 'depth') and x[0:periodInd] in fNamesBase] # make sure it's not a hidden file	
		fileNamesS = [x for x in fileNames if (x[-4:] == 'skel') and x[0:periodInd] in fNamesBase] # make sure it's not a hidden file		
		
		for j in [x for x in range(len(fNamesBase)) if np.mod(x,2) == 1]:

			fDelete = [x for x in fileNames if x[0:periodInd] == fNamesBase[j]]	
			if len(fDelete) > 0:
				for k in fDelete:
					print k
					os.remove(k)
#			os.remove(fileNamesD[i])
#			os.remove(fileNamesS[i])


