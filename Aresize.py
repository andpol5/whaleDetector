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

# resize all images to the average height and weight
h = 0
w = 0

for imageFileName in imageFileList:
    im = cv2.imread(imageFileName)
    w = w + im.shape[0]
    h = h + im.shape[1]
    
h = h / len(imageFileList)
w = w / len(imageFileList)

print "Average size: " + str(w) + " x " + str(h) + "\n"
h = h / 2
w = w / 2
print "Crop size: " + str(w) + " x " + str(h) + "\n"

# Create crop directory to move files to
if os.path.exists('crop'):
    shutil.rmtree('crop')
os.mkdir('crop')

for imageFileName in imageFileList:
    im = cv2.imread(imageFileName)
    imCrop = cv2.resize(im, (w, h))
    cv2.imwrite('crop/' + imageFileName, imCrop)

