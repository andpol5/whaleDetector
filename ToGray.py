# Move all the training files to training data directory
import os
import shutil
import numpy as np
import cv2

def getAllImageFiles():
    matches = []
    for root, dirnames, filenames in os.walk(os.getcwd()):
        filenames = [ f for f in filenames if os.path.splitext(f)[1] in ('.jpg') ]
        for filename in filenames:
            #matches.append(os.path.join(root, filename))
            matches.append(filename)
    return matches

imageFileList = getAllImageFiles()

# Create crop directory to move files to
if os.path.exists('gray'):
    shutil.rmtree('gray')
os.mkdir('gray')

for imageFileName in imageFileList:
    im = cv2.imread(imageFileName)
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray/' + imageFileName, imGray)

