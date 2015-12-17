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

# Whitening algorithm
def svd_whiten(X):
    U, s, Vt = np.linalg.svd(X)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white


imageFileList = getAllImageFiles()

# Create crop directory to move files to
if os.path.exists('new'):
    shutil.rmtree('new')
os.mkdir('new')

# resize all images to this height and weight
h = 128
w = 128

for imageFileName in imageFileList:
    im = cv2.imread(imageFileName)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im / 255.0        

    im = svd_whiten(im)* 255.0    
    #import ipdb; ipdb.set_trace()
    im = cv2.resize(im, (w, h))
    cv2.imwrite('new/' + imageFileName, im)

