#! /usr/bin/python
#
import os
import math
import sys
import shutil

import numpy as np
import cv2

noChangeMat = np.array([[1.,0.,0.],[0.,1.,0.]])
dir = 'alphaWhales/batch2'

def getAllImageFiles():
    matches = []
    for root, dirnames, filenames in os.walk(dir):
        filenames = [ f for f in filenames if os.path.splitext(f)[1] in ('.jpg') ]
        for filename in filenames:
            #matches.append(os.path.join(root, filename))
            matches.append(filename)
    return matches

# Rotate the image (CCW) about an angle (deg)
def rotate(img, angle):
    # Center of rotation is the center of the image
    center=tuple(np.array(img.shape[0:2])/2)
    # Affine transform - rotation about the center
    rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img,rotMat,img.shape[0:2],flags=cv2.INTER_LINEAR)

# Shift the image along x axis
def shiftx(img, deltax):
    shiftxMat = noChangeMat.copy()
    shiftxMat[0,2] = deltax
    return cv2.warpAffine(img,shiftxMat,img.shape[0:2],flags=cv2.INTER_LINEAR)

# Shear the image along x axis
# rx - shear factor
def shearx(img, rx):
    shearxMat = noChangeMat.copy()
    shearxMat[0,1] = rx
    return cv2.warpAffine(img,shearxMat,img.shape[0:2],flags=cv2.INTER_LINEAR)

# Shift the image along y axis
def shifty(img, deltay):
    shiftyMat = noChangeMat.copy()
    shiftyMat[1,2] = deltay
    return cv2.warpAffine(img,shiftyMat,img.shape[0:2],flags=cv2.INTER_LINEAR)

# Shear the image along x axis
# rx - shear factor
def sheary(img, ry):
    shearyMat = noChangeMat.copy()
    shearyMat[1,0] = ry
    return cv2.warpAffine(img,shearyMat,img.shape[0:2],flags=cv2.INTER_LINEAR)

imageFileList = getAllImageFiles()

deltax = 20
deltayPlus10 = 20
deltayPlus20 = 40
deltayMinus10 = -20
deltayMinus20 = -40
for imageFileName in imageFileList[1:]:
    im = cv2.imread(dir + "/" + imageFileName, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Extract "w_" from filename and write the new files
    name = imageFileName[:-4]
    cv2.imwrite(dir + "/" + name + '_1.jpg', shiftx(im, deltax))
    cv2.imwrite(dir + "/" + name + '_2.jpg', shifty(im, deltayPlus10))
    cv2.imwrite(dir + "/" + name + '_3.jpg', shifty(im, deltayPlus20))
    cv2.imwrite(dir + "/" + name + '_4.jpg', shifty(im, deltayMinus10))
    cv2.imwrite(dir + "/" + name + '_5.jpg', shifty(im, deltayMinus20))
