
# Remove all files from path with .rgb extension

import os


path = '/Users/colin/data/ICU_7March2012_Head/'
# playTypes = ['depth', 'rgb']
os.chdir(path)

dirNames = os.listdir('.')
dirNames = [x for x in dirNames if (x[0] != '.')] # make sure it's not a hidden file


# Get current directory
for i in range(len(dirNames)):
	os.chdir(path+dirNames[i])
	fileNames = os.listdir('.')
	fileNames = [x for x in fileNames if (x[-3:] == 'rgb')] # make sure it's not a hidden file
	for i in fileNames:
		print i
		os.remove(i)


