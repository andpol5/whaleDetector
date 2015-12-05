import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def getAllImageFiles()
    for root, dirnames, filenames in os.walk(source):
        filenames = [ f for f in filenames if os.path.splitext(f)[1] in ('.mov', '.MOV', '.avi', '.mpg') ]
        for filename in filenames:
            matches.append(os.path.join(root, filename))
    return matches

while True:
    image = cv.imread('w_90.jpg')

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

    # Filter the Hues channel again
    blurredH = cv.GaussianBlur(imageH, (11, 11), 0)

    # Binarize the hue channel image
    otsuThreshold, imageBinary = cv.threshold(blurredH, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Remove a 12 pixel border around the whole image to prevent dilation from taking over the border
    rectMask = np.zeros(imageBinary.shape[:2], np.uint8)
    x = y = 12
    h = imageBinary.shape[0] - 2*y
    w = imageBinary.shape[1] - 2*x
    rectMask[y:y+h, x:x+w] = 255
    # Combine the masks and invert
    imageBinary = cv.bitwise_and(~imageBinary, ~rectMask)

    # Erode and dialate to remove some of the noise
    imageBinary = cv.erode(imageBinary, None, iterations=3)
    imageBinary = cv.dilate(imageBinary, None, iterations=15)

    # Find contours in the resulting mask
    contours, hierarchy = cv.findContours(imageBinary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv.contourArea(c) for c in contours]
    maxContourIndex = np.argmax(areas)
    contour = contours[maxContourIndex]

    mask = np.zeros(imageBinary.shape, np.uint8)
    cv.drawContours(imageBinary,[contour],0,(0,255,0),2)
    cv.drawContours(mask,[contour],0,255,-1)

    # Find the bounding rectangle for the contour
    x, y, w, h = cv.boundingRect(contour)

    # Create the final cropped image
    croppedImage = cv.bitwise_and(image, cv.merge((mask, mask, mask)))
    croppedImage = croppedImage[y:y+h, x:x+w]

    cv.imshow('w', croppedImage)

    key = cv.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
	    break
cv.destroyAllWindows()

