#! /usr/bin/python
#
import os
import math
import sys
import shutil
import random

def getAllImageFiles():
    imageFileNames = []
    for root, dirnames, filenames in os.walk(os.getcwd()):
        filenames = [ f for f in filenames if os.path.splitext(f)[1] in ('.jpg') ]
        for filename in filenames:
            #matches.append(os.path.join(root, filename))
            imageFileNames.append(filename)
    return imageFileNames

def randomSample(list, percent):
    num = int(len(list)*percent)
    sample = [ list[i] for i in sorted(random.sample(xrange(len(list)), num)) ]
    remainder = [x for x in list if x not in sample]
    return [sample, remainder]

def moveFilesTo(filenames, dir):
    # Create crop directory to move files to
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    for file in filenames:
        shutil.copy(file, dir+'/'+file)

imageFileList = getAllImageFiles()

# Split 10% - test, and 90% remainder
testSet, remainder = randomSample(imageFileList, 0.1)

# Split the 90 into 15%-validation and 85%-train
validation, trainSet = randomSample(remainder, 0.15)

moveFilesTo(testSet, 'test')
moveFilesTo(validation, 'validation')
moveFilesTo(trainSet, 'train')
