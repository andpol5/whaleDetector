import numpy as np
import os
import re
import tensorflow as tf
from scipy import random
from PIL import Image
import time

import Dataset

# Tensorflow convinience functions
def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)

def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)

def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2dValidResize(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

start_time = time.time()
sess = tf.InteractiveSession()

# Constants 
nClasses = 3
imageSize = 128*128
batchSize = 37

# The size of the images is 256x256
x = tf.placeholder("float", shape=[None, imageSize])
# There are 4 classes (labels)
y_ = tf.placeholder("float", shape=[None, nClasses])

# CONVOLUTIONAL NEURAL NET
# The first two dimensions are the patch size, the next is the number of input channels, 
# and the last is the number of output channels. 
# We will also have a bias vector with a component for each output channel.
W_conv1 = weight_variable([5, 5, 1, 20])
b_conv1 = bias_variable([20])

# To apply the layer, we first reshape x to a 4d tensor, with the second 
# and third dimensions corresponding to image width and height, 
# and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1,128,128,1])

# We then convolve x_image with the weight tensor, 
# add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu6(conv2dValidResize(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# SECOND CONV LAYER
# In order to build a deep network, we stack several layers of this type. 
# The second layer will have 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 20, 50])
b_conv2 = bias_variable([50])

h_conv2 = tf.nn.relu6(conv2dValidResize(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# DENSELY CONNECTED LAYER
# Now that the image size has been reduced to 32x32, 
# we add a fully-connected layer with 1024 neurons to allow processing on the entire image. 
# We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight 
# matrix, add a bias, and apply a ReLU.

W_fc1 = weight_variable([29*29*50, 512])
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(h_pool2, [-1, 29*29*50])
h_fc1 = tf.nn.relu6(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# DROPOUT
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# READOUT LAYER
W_fc2 = weight_variable([512, nClasses])
b_fc2 = bias_variable([nClasses])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Load the dataset
dataset = Dataset.Dataset('imgs', 'whales.csv', sess)

# Train and eval the model
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
l2_loss = tf.nn.l2_loss(tf.clip_by_value(y_conv,1e-10,1.0))

#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(l2_loss)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(l2_loss)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in xrange(300):
   step_start = time.time()
   batch = dataset.get_batch(batchSize)
   labels = batch[1]

   # Format the Y for this step
   yTrain = np.zeros((batchSize, nClasses))
   for j in xrange(batchSize):
      yTrain[j][ int(labels[j]) ] = 1
   
   if i%5 == 0:
      train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: yTrain, keep_prob: 1.0}, session=sess)
      print "\n**step %d, training accuracy %g"%(i, train_accuracy)
               
   train_step.run(feed_dict={x: batch[0], y_: yTrain, keep_prob: 0.8}, session=sess)
#    if i%5 == 0:
#       ytmp = (y_conv.eval(feed_dict={x: batch[0], y_: yTrain, keep_prob: 1.0},session=sess))
#       import ipdb; ipdb.set_trace()
   print "\n***step %d finished, time = %s" %(i, time.time() - step_start)

# Evaluate the final accuracy
accuracies = []
for i in xrange(0, dataset.get_size(), batchSize):
   test = dataset.get_sequential_batch(batchSize, i)
   testLabels = test[1]
   
   returnBathSize = test[0].shape[0] 
   if returnBathSize > 0:
      yTest = np.zeros((returnBathSize, nClasses))
   #    import ipdb; ipdb.set_trace()
      for j in xrange(returnBathSize):
         yTest[j][ int(testLabels[j]) ] = 1
      
      accuracies.append(accuracy.eval(feed_dict={x: test[0], y_: yTest, keep_prob: 1.0},session=sess))
         
print "\n\n****Accuracy = " + str((sum(accuracies) / float(len(accuracies))))
print("--- %s seconds ---" % (time.time() - start_time))
