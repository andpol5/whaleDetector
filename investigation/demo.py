import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

while True:
    image = cv.imread('w_74.jpg')

    # Split by channel
    imageB, imageG, imageR = cv.split(image)

    # Calculate most occurring values in each channel, then use these
    # colors as filler for white areas in the original image. White pixels
    # are pixels with a value higher than TH.
    histR = cv.calcHist([image], [2], None, [256], [0,256])  
    histG = cv.calcHist([image], [1], None, [256], [0,256])
    histB = cv.calcHist([image], [0], None, [256], [0,256])

    # Get the index numbers of the peaks of each histogram       
    indexOfMaxR = np.argmax(histR) 
    indexOfMaxG = np.argmax(histG)
    indexOfMaxB = np.argmax(histB) 

    # Threshold for whiteness in the gray image
    WHITE_LEVEL_THRESHOLD = 175 
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imageR[imageGray > WHITE_LEVEL_THRESHOLD] = indexOfMaxR
    imageG[imageGray > WHITE_LEVEL_THRESHOLD] = indexOfMaxG
    imageB[imageGray > WHITE_LEVEL_THRESHOLD] = indexOfMaxB

    # Update the image channels to get an image without white areas
    updatedImage = cv.merge((imageB, imageG, imageR))

    # Gaussian filter
    blurred = cv.GaussianBlur(updatedImage, (11, 11), 0)

    # Convert to HSV
    imageHsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # Get the saturation channel and adjust its intensity
    imageH, imageS, imageV = cv.split(imageHsv)

    # Binarize the saturation image
    newThresh, thresholdMatrix = cv.threshold(imageS, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # 0.7 is adjustment to get the TH slightly smaller than the global value
    newThresh = newThresh * 0.7 
    nt, imageBinary = cv.threshold(imageS, newThresh, 255, cv.THRESH_BINARY)

    # Erode and dialate to remove some of the noise
    imageBinary = cv.erode(imageBinary, None, iterations=10)
    imageBinary = cv.dilate(imageBinary, None, iterations=20)

    # Find contours in the resulting mask
    contours, hierarchy = cv.findContours(imageBinary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv.contourArea(c) for c in contours]
    maxContourIndex = np.argmax(areas)
    contour = contours[maxContourIndex]

    mask = np.zeros(imageBinary.shape, np.uint8)
    cv.drawContours(imageBinary,[contour],0,(0,255,0),2)
    cv.drawContours(mask,[contour],0,255,-1)

    imageHBlurred = cv.GaussianBlur(imageH, (11, 11), 0)
    cv.imshow('w', imageHBlurred)

    histr = cv.calcHist([imageHBlurred],[0],None,[256],[0,256])
    plt.plot(histr)
    plt.xlim([0,256])
    plt.show()

    key = cv.waitKey(1) & 0xFF
    
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
	    break
cv.destroyAllWindows()

