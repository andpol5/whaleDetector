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
imageSize = 256*256
batchSize = 20

# The size of the images is 200x150
x = tf.placeholder("float", shape=[None, imageSize])
# There are 4 classes (labels)
y_ = tf.placeholder("float", shape=[None, nClasses])

# CONVOLUTIONAL NEURAL NET
# The first two dimensions are the patch size, the next is the number of input channels, 
# and the last is the number of output channels. 
# We will also have a bias vector with a component for each output channel.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 4d tensor, with the second 
# and third dimensions corresponding to image width and height, 
# and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1,256,256,1])

# We then convolve x_image with the weight tensor, 
# add the bias, apply the ReLU function, and finally max pool.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# SECOND CONV LAYER
# In order to build a deep network, we stack several layers of this type. 
# The second layer will have 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# DENSELY CONNECTED LAYER
# Now that the image size has been reduced to 7x7, 
# we add a fully-connected layer with 1024 neurons to allow processing on the entire image. 
# We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight 
# matrix, add a bias, and apply a ReLU.

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# DROPOUT
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# READOUT LAYER
W_fc2 = weight_variable([1024, nClasses])
b_fc2 = bias_variable([nClasses])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Load the dataset
dataset = Dataset('jack_jill_imgs', 'jackAndJill.csv')

# Train and eval the model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in xrange(100):
   batch = dataset.get_batch(batchSize)
   labels = batch[1]

   # Format the Y for this step
   # I don't know why it only recognizes 18 out of 19 images   
   yTrain = np.zeros((batchSize, nClasses))
   for j in xrange(batchSize):
#       import ipdb; ipdb.set_trace()
      yTrain[j][ int(labels[j]) ] = 1

   yTrain = yTrain.tolist()
   
   if i%5 == 0:
      import ipdb; ipdb.set_trace()
      train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: yTrain, keep_prob: 1.0})
      print "step %d, training accuracy %g"%(i, train_accuracy)

   train_step.run(feed_dict={x: batch[0], y_: yTrain, keep_prob: 0.5}, session=sess)

# Evaluate the prediction
test = dataset.get_batch(batchSize)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

yTest = np.zeros((batchSize, nClasses))
for j in xrange(batchSize):
   yTest[j][ int(labels[j]) ] = 1

print "Accuracy = "
print accuracy.eval(feed_dict={x: test[0], y_: yTest})
print("--- %s seconds ---" % (time.time() - start_time))
