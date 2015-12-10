# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image

from collections import defaultdict
import re
from scipy import random
import tensorflow as tf
import cv2


class DataSet(object):

  def __init__(self, inputDir, labelsFile, fake_data=False, one_hot=False):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""
    self.indexInEpoch = 0
    self.epochsCompleted = 0
    self.inputDir = inputDir
    output = np.genfromtxt(labelsFile, skip_header=1, dtype=[('image', 'S10'), ('label', 'S11')], delimiter=',')

    labels = [x[1] for x in output]

    self.numberOfClasses = len(set(labels))
    self.images = []
    self.sizes = []
    for file in os.listdir(inputDir):
       if(file.endswith('.jpg')):
          self.images.append(file)
          im = Image.open(os.path.join(inputDir, file))
          w,h = im.size
          self.sizes.append((w,h))
    self.images = np.array(self.images)

    maxSize = np.min(self.sizes,axis=0)
    self.maxW = int(maxSize[0])
    self.maxH = int(maxSize[1])
    self.maxSize = int(self.maxW * self.maxH)

    self.labelsDict = {int(re.search("w_(\\d+)\.jpg", x[0]).group(1)):int(re.search("whale_(\\d+)", x[1]).group(1)) for x in output}

    # to provide default values
    self.labelsDict = defaultdict(lambda:0, self.labelsDict)

    self.examples = [int(re.search("w_(\\d+).jpg",x).group(1)) for x in self.images]
    self.labels = [self.labelsDict[x] for x in self.examples]
    self.labels = np.array(self.labels)

    self.imagesIds = np.array(self.examples)
    self.train = self.images[self.labels > 0]
    self.origTrainLabels = self.labels[self.labels > 0]

    l = np.zeros(max(self.origTrainLabels))
    i = 0
    for k in sorted(set(self.origTrainLabels)):
       l[k-1] = i
       i += 1
    self.trainLabels = np.array([l[x-1] for x in self.origTrainLabels])
    self.test = self.images[self.labels == 0]

  @property
  def num_examples(self):
    return len(self.train)

  @property
  def epochs_completed(self):
    return self.epochsCompleted


  def next_batch(self, batchSize, fake_data=False):
    start = self.indexInEpoch
    self.indexInEpoch += batchSize
    if self.indexInEpoch > self.num_examples:
       # Finished epoch
       self.epochsCompleted += 1
       # Shuffle the data
       perm = np.arange(self.num_examples)
       np.random.shuffle(perm)
       self.train = self.train[perm]
       self.trainLabels = self.trainLabels[perm]
       # Start next epoch
       start = 0
       self.indexInEpoch = batchSize
       assert batchSize <= self.num_examples
    end = self.indexInEpoch
    return self.read_images([os.path.join(self.inputDir, x) for x in self.train[start:end]]), self.trainLabels[start:end]

  def read_images(self, filenames):
    images = []

    for file in filenames:
      im = cv2.imread(file)[:,:,0]
      images.append(im.flatten())
    return np.asarray(images)


def read_data_sets(trainDir, labelsFile):
  class DataSets(object):
    pass
  data_sets = DataSets()

  data_sets.train = DataSet(trainDir, labelsFile)
  data_sets.validation = DataSet(trainDir, labelsFile)
  data_sets.test = DataSet(trainDir, labelsFile)

  return data_sets
