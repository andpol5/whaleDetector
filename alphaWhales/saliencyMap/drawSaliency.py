import numpy as np
import cv2

def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))*255
whale = 'w_38'
im = cv2.imread('../train/train/%s.jpg' % whale)

salIm = cv2.imread('%s/salience_map.jpg'%whale,flags=cv2.IMREAD_GRAYSCALE)
salImNew = np.zeros((im.shape[0], im.shape[1]))
rectSize = 8
for i in xrange(salImNew.shape[0]/rectSize):
    for j in xrange(salImNew.shape[1]/rectSize):
        salImNew[i*rectSize:(i+1)*rectSize, j*rectSize:(j+1)*rectSize] = np.ones((rectSize, rectSize))*salIm[i,j]
#imMerged = im/float(2)
imMerged = cv2.applyColorMap(np.uint8(salImNew), cv2.COLORMAP_JET)
#imMerged[:,:,0] = imMerged[:,:,0] + (salImNew/float(4))
#imMerged[:,:,0] = normalize(imMerged[:,:,0])
#imMerged[:,:,2] = imMerged[:,:,2] + 112-(salImNew/float(4))
#imMerged[:,:,2] = normalize(imMerged[:,:,2])
imMerged = normalize(im+(imMerged))
cv2.imwrite('%s/merged.jpg'%whale, imMerged)