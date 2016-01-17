from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from collections import defaultdict
import re
import cv2
from scipy import random

# Whitening algorithm
def svd_whiten(X):
    U, s, Vt = np.linalg.svd(X)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white

class DataSet(object):

   def __init__(self, inputDir, labelsFile, nClasses = 38):
      self.indexInEpoch = 0
      self.epochsCompleted = 0

      self.inputDir = inputDir
      output = np.genfromtxt(labelsFile, skip_header=1, dtype=[('image', 'S13'), ('label', 'S11')], delimiter=',')

      self.images = []

      for file in os.listdir(inputDir):
         if (file.endswith('.jpg')):
            self.images.append(file)

      self.allImages = [x[0] for x in output]
      self.allLabels = [int(re.search("whale_(\\d+)", x[1]).group(1)) for x in output]

      # sorted list of unique whale ids
      self.classes = sorted(set(self.allLabels))
      self.numberOfClasses = len(set(self.classes))
      assert self.numberOfClasses == 38

      self.labels = []
      for file in self.images:
         ind = self.allImages.index(file)
         cl = self.allLabels[ind]
         # assign a class from 0 to 37
         newClass = self.classes.index(cl)
         self.labels.append(newClass)

      # Format the Y for this step
      self.yTrain = np.zeros((len(self.labels), nClasses))
      for j in xrange(len(self.labels)):
         self.yTrain[j][self.labels[j]] = 1
      self.images = np.array(self.images)

   @property
   def num_examples(self):
      return len(self.images)

   @property
   def epochs_completed(self):
      return self.epochsCompleted

   def getAll(self):
         return self.read_images([os.path.join(self.inputDir, x) for x in self.images]), self.yTrain

   def get_random_batch(self, batchSize):
      randInd = random.permutation(len(self.images))[:batchSize]
      return self.read_images([os.path.join(self.inputDir, x) for x in self.images[randInd]]), self.yTrain[randInd]

   def get_sequential_batch(self, batchSize):
      start = self.indexInEpoch
      self.indexInEpoch += batchSize
      if self.indexInEpoch > self.num_examples:
         # Finished epoch
         self.epochsCompleted += 1
         # Shuffle the data
         perm = np.arange(self.num_examples)
         np.random.shuffle(perm)
         self.images = self.images[perm]
         self.yTrain = self.yTrain[perm]
         # Start next epoch
         start = 0
         self.indexInEpoch = batchSize
         assert batchSize <= self.num_examples
      end = self.indexInEpoch
      return self.read_images([os.path.join(self.inputDir, x) for x in self.images[start:end]]), self.yTrain[start:end], self.images[start:end]

   def read_images(self, filenames):
      images = []

      for file in filenames:
         im = cv2.imread(file, flags=cv2.IMREAD_GRAYSCALE)
         normIm = im.astype(float)

         normIm = (normIm/255)
         # normIm = cv2.resize(normIm, (227,227))
         images.append(np.reshape(normIm, (normIm.shape[0], normIm.shape[1], 1)))
      return np.asarray(images)

def read_data_sets(trainDir, validationDir, trainFile, validationFile):
   class DataSets(object):
      pass

   data_sets = DataSets()

   data_sets.train = DataSet(trainDir, trainFile)
   data_sets.validation = DataSet(validationDir, validationFile)

   return data_sets
