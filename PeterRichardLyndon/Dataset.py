import numpy as np
import os
import re
import tensorflow as tf
from scipy import random
from PIL import Image


from collections import defaultdict

class Dataset(object):
   def __init__(self, inputDir, labelsFile, tfSession):
      self.inputDir = inputDir
      self.tfSession = tfSession
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
   
   def get_batch(self, size):
      randInd = random.permutation(len(self.train))[:size]
      return self.read_images([os.path.join(self.inputDir, x) for x in self.train[randInd]]), self.trainLabels[randInd]

   # Return size of dataset
   def get_size(self):
      return self.images.shape[0]

   def get_sequential_batch(self, batchSize, startIndex):
      endIndex = startIndex + batchSize;
      if endIndex > self.get_size()-1:
         endIndex = -1
      slicedPart = range(startIndex,endIndex)
      return self.read_images([os.path.join(self.inputDir, x) for x in self.train[slicedPart]]), self.trainLabels[slicedPart]

   # TODO fix the reader to not throw up enqueue errors
   def read_images(self, filenames):
      images = []
      reader = tf.WholeFileReader()
      
      if len(filenames) > 0:
         jpeg_file_queue = tf.train.string_input_producer(filenames)
         jkey, jvalue = reader.read(jpeg_file_queue)
         j_img = tf.image.decode_jpeg(jvalue)
      
      with tf.Session() as sess:
         # Start populating the filename queue.
         coord = tf.train.Coordinator()
         threads = tf.train.start_queue_runners(coord=coord)
         
         if len(filenames) > 0:
            for i in range(len(filenames)):
               jpeg = j_img.eval()
               images.append(jpeg.flatten())
         
         coord.request_stop()
         coord.join(threads)
      
      return np.asarray(images)
