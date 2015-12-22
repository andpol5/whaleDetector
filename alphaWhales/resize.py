import os
import cv2

filenames = []
indir = 'batch2'
for file in os.listdir(indir):
   if(file.endswith('.jpg')):
      filenames.append(file)

for file in filenames:
   im = cv2.imread("batch2/"+file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
   crop_img = im[14:241, 14:241]
   cv2.imwrite("batch2/"+file, crop_img)
