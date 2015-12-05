# Whale Detection preprocessor
# detects the ROI of each whale in an image

import os
import numpy as np
import cv2

def whaleExport(readDir, writeDir):
    # Get all files in readDir
    files = []
    for (dirpath, dirnames, filenames) in os.walk(readDir):
        files.extend(filenames)
        break

    # Loop through the images in 'readDir'
    for fileName in files:
        if not os.path.isfile(readDir + '/' + fileName):
            continue

        # Read an image
        image = cv2.imread(readDir + '/' + fileName);
        newFileName = writeDir + '/' + fileName

        # Split by channel
        imageB, imageG, imageR = cv2.split(image)

        # Calculate most occurring values in each channel, then use these
        # colors as filler for white areas in the original image. White pixels
        # are pixels with a value higher than TH.
        histR = cv2.calcHist([image], [2], None, [256], [0,256])  
        histG = cv2.calcHist([image], [1], None, [256], [0,256])
        histB = cv2.calcHist([image], [0], None, [256], [0,256])

        # Get the index numbers of the peaks of each histogram       
        indexOfMaxR = np.argmax(histR) 
        indexOfMaxG = np.argmax(histG)
        indexOfMaxB = np.argmax(histB) 

        # Threshold for whiteness in the gray image
        WHITE_LEVEL_THRESHOLD = 175 
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageR[imageGray > WHITE_LEVEL_THRESHOLD] = indexOfMaxR
        imageG[imageGray > WHITE_LEVEL_THRESHOLD] = indexOfMaxG
        imageB[imageGray > WHITE_LEVEL_THRESHOLD] = indexOfMaxB

        # Update the image channels to get an image without white areas
        updatedImage = cv2.merge((imageB, imageG, imageR))

        # Gaussian filter
        blurred = cv2.GaussianBlur(updatedImage, (11, 11), 0)

        # Convert to HSV
        imageHsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Get the saturation channel and adjust its intensity
        imageH, imageS, imageV = cv2.split(imageHsv)

        # Filter the saturation channel again
        blurredS = cv2.GaussianBlur(imageS, (11, 11), 0)
        cv2.imwrite(newFileName + '_sat', blurredS)

        # Filter the saturation channel again
        blurredH = cv2.GaussianBlur(imageH, (11, 11), 0)
        cv2.imwrite(newFileName + '_hue', blurredH)

        # Binarize the hue channel image
        otsuThreshold, imageBinary = cv2.threshold(blurredS, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Remove a 12 pixel border around the whole image to prevent dilation from taking over
        rectMask = np.zeros(imageBinary.shape[:2], np.uint8)
        x = y = 12
        h = imageBinary.shape[0] - 2*y
        w = imageBinary.shape[1] - 2*x
        rectMask[y:y+h, x:x+w] = 255
        # Combine the masks and invert 
        imageBinary = cv2.bitwise_and(~imageBinary, ~rectMask)

        # Erode and dialate to remove some of the noise
        imageBinary = cv2.erode(imageBinary, None, iterations=3)
        imageBinary = cv2.dilate(imageBinary, None, iterations=15)
        cv2.imwrite(newFileName + '_binary', imageBinary)

        # Find contours in the resulting mask
#        contours, hierarchy = cv2.findContours(imageBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the index of the largest contour
#        areas = [cv2.contourArea(c) for c in contours]
#        maxContourIndex = np.argmax(areas)
#        contour = contours[maxContourIndex]

#        mask = np.zeros(imageBinary.shape, np.uint8)
#        cv2.drawContours(imageBinary,[contour],0,(0,255,0),2)
#        cv2.drawContours(mask,[contour],0,255,-1)
#        cv2.imwrite(newFileName + '_mask', mask)

        # Find the bounding rectangle for the contour
#        x, y, w, h = cv2.boundingRect(contour)

        # Create the final cropped image
#        croppedImage = cv2.bitwise_and(image, cv2.merge((mask, mask, mask)))
#        croppedImage = croppedImage[y:y+h, x:x+w]

        # Save image
#        cv2.imwrite(newFileName, croppedImage)

