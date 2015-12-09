import numpy as np
import os
import re
import tensorflow as tf
from scipy import random
from PIL import Image
import cv2
import time


from collections import defaultdict

class Dataset(object):
   def __init__(self, inputDir, labelsFile):
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
   
   def get_batch(self, size):
      randInd = random.permutation(len(self.train))[:size]
      return self.read_images([os.path.join(self.inputDir, x) for x in self.train[randInd]]), self.trainLabels[randInd]

   # Return size of dataset
   def get_size(self):
      return len(self.train)

   def get_sequential_batch(self, batchSize):
      start = self.indexInEpoch
      self.indexInEpoch += batchSize
      if self.indexInEpoch > self.get_size():
         # Finished epoch
         self.epochsCompleted += 1
         # Shuffle the data
         perm = np.arange(self.get_size())
         np.random.shuffle(perm)
         self.train = self.train[perm]
         self.trainLabels = self.trainLabels[perm]
         # Start next epoch
         start = 0
         self.indexInEpoch = batchSize
         assert batchSize <= self.get_size()
      end = self.indexInEpoch
      return self.read_images([os.path.join(self.inputDir, x) for x in self.train[start:end]]), self.trainLabels[start:end]

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

# Tensorflow convinience functions
def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)

def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

start_time = time.time()
sess = tf.InteractiveSession()

# Constants 
nClasses = 2
imageSize = 128*128
batchSize = 50

# The size of the images is 200x150
x = tf.placeholder("float", shape=[None, imageSize])
# There are 4 classes (labels)
y_ = tf.placeholder("float", shape=[None, nClasses])

# CONVOLUTIONAL NEURAL NET
# The first two dimensions are the patch size, the next is the number of input channels, 
# and the last is the number of output channels. 
# We will also have a bias vector with a component for each output channel.
d1 = 32
d2 = 32
d3 = 64
d4 = 64
d5 = 300
W_conv1 = weight_variable([5, 5, 1, d1])
b_conv1 = bias_variable([d1])

# To apply the layer, we first reshape x to a 4d tensor, with the second 
# and third dimensions corresponding to image width and height, 
# and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1,128,128,1])

# We then convolve x_image with the weight tensor, 
# add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# SECOND CONV LAYER
# In order to build a deep network, we stack several layers of this type. 
# The second layer will have 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, d1, d2])
b_conv2 = bias_variable([d2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# THIRD CONV LAYER
W_conv3 = weight_variable([5, 5, d2, d3])
b_conv3 = bias_variable([d3])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# FORTH CONV LAYER
W_conv4 = weight_variable([3, 3, d3, d4])
b_conv4 = bias_variable([d4])

h_conv4 = tf.nn.relu6(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)


# DENSELY CONNECTED LAYER
# Now that the image size has been reduced to 7x7, 
# we add a fully-connected layer with 1024 neurons to allow processing on the entire image. 
# We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight 
# matrix, add a bias, and apply a ReLU.

W_fc1 = weight_variable([8*8*d4, d5])
b_fc1 = bias_variable([d5])

h_pool4_flat = tf.reshape(h_pool4, [-1, 8*8*d4])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

# DROPOUT
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# READOUT LAYER
W_fc2 = weight_variable([d5, nClasses])
b_fc2 = bias_variable([nClasses])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Load the dataset
dataset = Dataset('jack_jill_imgs', 'jackAndJill.csv')

# Train and eval the model
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))

# train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

f1=open('log', 'w+')
f1.write('AAAAA\n')
f1.write("Start %s\n" % time.time())
f1.flush()

for i in xrange(100):
   step_start = time.time()
   
   if i%10 == 0 and i!=0:
      test = dataset.get_batch(80)
      testLabels = test[1]

      yTest = np.zeros((80, nClasses))
      for j in xrange(80):
         yTest[j][ int(testLabels[j]) ] = 1

      f1.write("Accuracy = \n")
      f1.write(str(accuracy.eval(feed_dict={x: test[0], y_: yTest, keep_prob: 1.0}))+'\n')
      f1.flush()

   batch = dataset.get_sequential_batch(batchSize)
   labels = batch[1]

   # Format the Y for this step
   yTrain = np.zeros((batchSize, nClasses))
   for j in xrange(batchSize):
      yTrain[j][ int(labels[j]) ] = 1
   train_step.run(feed_dict={x: batch[0], y_: yTrain, keep_prob: 0.5}, session=sess)

   f1.write("step %d finished, time = %s\n" %(i, time.time() - step_start))
   f1.write(str(cross_entropy.eval(feed_dict={x: batch[0], y_: yTrain, keep_prob: 1}, session=sess))+"\n")
   f1.write(str(np.argmax(y_conv.eval(feed_dict={x: batch[0], y_: yTrain, keep_prob: 1}, session=sess), axis=1))+"\n")
   f1.flush()

# Evaluate the prediction
test = dataset.get_batch(80)
testLabels = test[1]

yTest = np.zeros((80, nClasses))
for j in xrange(80):
   yTest[j][ int(testLabels[j]) ] = 1

f1.write("Accuracy = \n")
f1.write(str(accuracy.eval(feed_dict={x: test[0], y_: yTest, keep_prob: 1.0})))
f1.write("\n--- %s seconds ---" % (time.time() - start_time))
f1.flush()
f1.close()
