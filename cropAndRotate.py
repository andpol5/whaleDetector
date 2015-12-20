#! /usr/bin/python
#
import os
import math
import sys

import numpy as np
import cv2

#Get the histogram. All histograms must be the same kind
def getHist(im):
    hist = cv2.calcHist([im],[0,1,2],None,[16,16,16],[0,255,0,255,0,255])
    return(hist)

#Divide the image in four subimages
def divImage(im):
    height,width,channels = im.shape
    im1 = im[0:height/2, 0:width/2]
    im2 = im[0:height/2, width/2:width]
    im3 = im[height/2:height, 0:width/2]
    im4 = im[height/2:height, width/2:width]
    return([im1,im2,im3,im4])

#Sets a whole image to a value. By default, black
def setImage(im,val=0):
    for i in range(0,3):
        im[:,:,i]=val

#Create a mask image to select regions with a distinct histogram.
#  white zones are different enough from the base image
#  im: image
#  baseHist: Histogram of the base image (currently, the original image)
#  label: A label for the current step, just for debug
def simHist(im, baseHist, label=""):
    #print label
    height,width,channels = im.shape
    if width < 10 or height < 10:
        setImage(im,0)
        return
    images = divImage(im)
    histg = baseHist
    #Test: this compares each subimage with the current image's histogram instead of the base (whole image)
    #histg = getHist(im)
    # CORREL: Other similarity measures may be tested
    sim = [cv2.compareHist(histg, getHist(imx), cv2.HISTCMP_CORREL) for imx in images]

    for i in range(0,len(sim)):
        #This is the threshold to consider too different
        if sim[i] < 0.25:
            setImage( images[i], 255 )
        else:
            sim.append( simHist( images[i], histg, label+str(i) ) )
    return(sim)

def intround(floatNum):
    return int(round(floatNum))

#### Draw the ellipse major and minor axis
def drawEllipseAxis(image, ellipse):
    # Center of ellipse
    cx = intround(ellipse[0][0])
    cy = intround(ellipse[0][1])

    # Get the half lengths of the two axes
    mjAxis = max(ellipse[1])
    miAxis = min(ellipse[1])

    miAngle = (ellipse[2] * math.pi / 180.0)
    mjAngle = miAngle + math.pi / 2.0

    mjAxis = max(ellipse[1]) / 2.0
    miAxis = min(ellipse[1]) / 2.0

    # major axis
    mj1x = intround(cx + mjAxis * math.cos(mjAngle))
    mj1y = intround(cy + mjAxis * math.sin(mjAngle))
    mj2x = intround(cx + mjAxis * math.cos(mjAngle + math.pi))
    mj2y = intround(cy + mjAxis * math.sin(mjAngle + math.pi))
    cv2.line(image, (mj1x, mj1y), (mj2x, mj2y),(255,0,0),2)

    # minor axis
    mi1x = intround(cx + miAxis * math.cos(miAngle))
    mi1y = intround(cy + miAxis * math.sin(miAngle))
    mi2x = intround(cx + miAxis * math.cos(miAngle + math.pi))
    mi2y = intround(cy + miAxis * math.sin(miAngle + math.pi))
    cv2.line(image, (mi1x, mi1y), (mi2x, mi2y),(255,0,0),2)

def rotateAndCrop(img, ellipse):
    # Center of rotation is the center of the image
    center=tuple(np.array(img.shape[0:2])/2)

    # Affine transform - rotation about the center
    rotMat = cv2.getRotationMatrix2D(center, ellipse[2], 1.0)

    # Recalculate the size of the rotated image
    # so the corners do not get cut off
    (h, w) = img.shape[0:2]
    r = np.deg2rad(ellipse[2])
    w, h = (abs(np.sin(r)*h) + abs(np.cos(r)*w),abs(np.sin(r)*w) + abs(np.cos(r)*h))

    rotated = cv2.warpAffine(img,rotMat,dsize=(int(w),int(h)),flags=cv2.INTER_LINEAR)

    # Get the center of the ellipse and rotate it with the affine transform
    centEll = np.ones((3,1))
    centEll[0] = ellipse[0][0]
    centEll[1] = ellipse[0][1]
    centEll = np.dot(rotMat, centEll)

    # Get the half lengths of the two axes
    mjAxis = max(ellipse[1]) / 2.0
    miAxis = min(ellipse[1]) / 2.0

    p1x = intround(centEll[0] - miAxis)
    p1y = intround(centEll[1] - mjAxis)
    p2x = intround(centEll[0] + miAxis)
    p2y = intround(centEll[1] + mjAxis)

    if p1x < 0:
        p1x = 0
    if p1y < 0:
        p1y = 0
    if p2x >= img.shape[1]:
        p2x = img.shape[1]-1
    if p2y >= img.shape[0]:
        p2y = img.shape[0]-1
    # Return a slice as the cropped ROI
    return rotated[p1y:p2y, p1x:p2x]

def processWhaleImage(img):
    #PREPROCESS
    #   Convert to HSV, yields better results than BGR
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #   Testing if CLAHE improves result
    #im = clahe(im)
    #   get original image hist
    baseHist = getHist(hsvImage)

    #PROCESS
    #   Perform region selection by histogram similarity. Returns a mask on whole image
    res = simHist(hsvImage,baseHist)

    #POSTPROCESS
    im = cv2.cvtColor(hsvImage, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im, 127, 255, 0)

    _,contours,hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_size = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        size = w * h
        if size > best_size:
            best_size = size
            best = cnt
        cv2.rectangle(im,(x-10,y-10),(x+w+10,y+h+10),(255,255,255),2)

    #Select biggest contour, assume that's the whale :D
    #  shows boundng box and contour
    if best_size > 0:
        # Fit an ellipse into this contour
        ellipse = cv2.fitEllipse(best)
    else:
        print "Could not find extract in " + fname

    # cv2.ellipse(org, ellipse, (0,255,0),2)
    # drawEllipseAxis(org, ellipse)
    return rotateAndCrop(img, ellipse)

#Crop all the whale images in a folder into a different folder
def whaleExport(readDir, writeDir):
    # Get all files in readDir
    files = []
    for (dirpath, dirnames, filenames) in os.walk(readDir):
        files.extend(filenames)
        break

    # Loop through the images in 'readDir'
    for fname in files:
        if not os.path.isfile(readDir + '/' + fname):
            continue
        fpath = readDir + '/' + fname
        im = cv2.imread(fpath)
        if im is None:
            print 'Failed to load image file:', fname
            sys.exit(1)
        # try:
        proccessed = processWhaleImage(im)
        cv2.imwrite(writeDir + '/' + fname, proccessed)
        # except:
        #     print(fname+" did not work")

#MAIN
readPath = sys.argv[1]
writePath = sys.argv[2]
whaleExport(readPath, writePath)
