# Whale Detection preprocessor
# detects the ROI of each whale in an image

from os import walk
import numpy as np
import cv2

def whaleExport(readDir, writeDir)
   # Get all files in readDir
   files = []
   for (dirpath, dirnames, filenames) in walk(mypath):
       f.extend(filenames)
       break

   # Loop through the images in 'readDir'
   for file in files:
      # Read an image
      image = cv2.imread(file);
      
      # Scale down the image
      #scale=0.4;
      #im=cv2.imresize(I,scale)
      
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
      imageGray = cv2.cvtColor(image, cv2.CV_RGB2GRAY)
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
      
      # Binarize the saturation image
      newThresh, thresholdMatrix = cv2.threshold(imageS, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      # 0.7 is adjustment to get the TH slightly smaller than the global value
      newThresh = newThresh * 0.7 
      imageBinary = cv2.threshold(imageS, 0, newThresh, cv2.THRESH_BINARY)

      # Erode and dialate to remove some of the noise
      imageBinary = cv2.erode(imageBinary, None, iterations=10)
      imageBinary = cv2.dilate(imageBinary, None, iterations=20)
      
      # Find contours in the resulting mask
      contours, hierarchy = cv2.findContours(imageBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      
      # Find the index of the largest contour
      areas = [cv2.contourArea(c) for c in contours]
      maxContourIndex = np.argmax(areas)
      contour = contours[maxContourIndex]
            
      # Find areas in the binary image and return the largest area, which
      # will be the whale
      mask = np.zeros(imageBinary.shape, np.uint8)
      cv2.drawContours(imageBinary,[contour],0,(0,255,0),2)
      cv2.drawContours(mask,[contour],0,255,-1)

      #### PYTHON DONE UP TO HERE
        
      # Remove other areas from the image, except the whale
      imBin(:,:)=0
      imBin(CC.PixelIdxList{idx})=1
      
      # Calculate the convex hull of the area
      imCH=bwconvhull(imBin,'objects',8)
      
      # Scale back the convex hull to the original size of the image
      imCH=imresize(imCH,[size(I,1),size(I,2)])
      
      # Calculate the properties of the convex hull of the whale
      # Extrema - will be used to define the cropping dimensions
      # 'MajorAxisLength' and 'MinorAxisLength' - will be used to calculate
      # the ratio of the convex hull. If the ratio is less than 2, the convex
      # either contains a lot of nearby waters OR the area of the visible
      # whale is very small. In both cases, this image is higly likely to 
      # to be detrimental to the neural network training.
      RP=regionprops(imCH,'Extrema','MajorAxisLength','MinorAxisLength')
      axisRatio=RP.MajorAxisLength/RP.MinorAxisLength
      
      # Check the ratio
      if axisRatio>2:
         cropX=RP.Extrema(8,1)
         cropY=RP.Extrema(1,2)
         cropW=RP.Extrema(3,1)-cropX
         cropH=RP.Extrema(5,2)-cropY
         # Extract only the whale from the original image
         imDet=I.*cast(repmat(imCH,[1 1 3]),'like',I)
         
         # Crop the image to the size of the convex hull
         imCropped=imcrop(imDet,[cropX cropY cropW cropH])
         
         #Save images
         fileName = [writeDir,'/',sprintf(formatStr,i)]
         imwrite(imCropped,fileName)
