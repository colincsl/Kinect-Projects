import os
import sys, shutil

mvDir = '/Users/colin/data/ICU_7May2012_Wide/'
os.chdir('/Volumes/ICU/ICUdata/')
dirs = os.listdir('.')
dirs = [x for x in dirs if x[0]!="."]

for i in dirs:
	os.chdir(i)
	files = os.listdir('.')
	files = [x for x in files if x[-5:]=="depth"]
	os.mkdir(mvDir+i)
	for j in files:
		shutil.copy(j, mvDir+i)
	os.chdir('..')
