from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from collections import defaultdict
import re
import cv2
from scipy import random


class DataSet(object):

   def __init__(self, inputDir, labelsFile):
      self.indexInEpoch = 0
      self.epochsCompleted = 0

      self.inputDir = inputDir
      output = np.genfromtxt(labelsFile, skip_header=1, dtype=[('image', 'S13'), ('label', 'S11')], delimiter=',')

      labels = [x[1] for x in output]

      self.numberOfClasses = len(set(labels))
      self.filenames = []

      for file in os.listdir(inputDir):
         if (file.endswith('.jpg')):
            self.filenames.append(file)

      self.filenames = np.array(self.filenames)
      self.labelsDict = {re.search("w_(\\w+)\.jpg", x[0]).group(1): int(re.search("whale_(\\d+)", x[1]).group(1))
                         for x in output}

      # to provide default values
      self.labelsDict = defaultdict(lambda: 0, self.labelsDict)

      self.examples = [re.search("w_(\\w+).jpg", x).group(1) for x in self.filenames]
      self.allLabels = [self.labelsDict[x] for x in self.examples]
      self.allLabels = np.array(self.allLabels)

      self.imagesIds = np.array(self.examples)
      self.images = self.filenames[self.allLabels > 0]
      self.origLabels = self.allLabels[self.allLabels > 0]

      l = np.zeros(max(self.origLabels))
      i = 0
      for k in sorted(set(self.origLabels)):
         l[k - 1] = i
         i += 1
      self.labels = np.array([l[x - 1] for x in self.origLabels])

   @property
   def num_examples(self):
      return len(self.images)

   @property
   def epochs_completed(self):
      return self.epochsCompleted

   def getAll(self):
         return self.read_images([os.path.join(self.inputDir, x) for x in self.images]), self.labels

   def get_random_batch(self, batchSize):
      randInd = random.permutation(len(self.images))[:batchSize]
      return self.read_images([os.path.join(self.inputDir, x) for x in self.images[randInd]]), self.labels[randInd]

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
         self.labels = self.labels[perm]
         # Start next epoch
         start = 0
         self.indexInEpoch = batchSize
         assert batchSize <= self.num_examples
      end = self.indexInEpoch
      return self.read_images([os.path.join(self.inputDir, x) for x in self.images[start:end]]), self.labels[start:end]

   def read_images(self, filenames):
      images = []

      for file in filenames:
         im = cv2.imread(file, flags=cv2.IMREAD_GRAYSCALE)
         normIm = im.astype(float)
         normIm = (normIm/np.max(normIm))*2.0-1.0
         images.append(normIm.flatten())
      return np.asarray(images)

def read_data_sets(trainDir, validationDir, trainFile, validationFile):
   class DataSets(object):
      pass

   data_sets = DataSets()

   data_sets.train = DataSet(trainDir, trainFile)
   data_sets.validation = DataSet(validationDir, validationFile)

   return data_sets
