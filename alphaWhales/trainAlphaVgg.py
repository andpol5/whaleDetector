#! /usr/bin/python
import numpy as np
import tensorflow as tf
import time
import dataset

f1=open('log_%d' % time.time(), 'w+')
f1.write('AAAAA\n')
f1.write("Start %s\n" % time.time())
f1.flush()

# Tensorflow convinience functions
def weight_variable(shape, name):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial, name=name)

def bias_variable(shape, name):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial, name=name)

def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


start_time = time.time()
######################################################################
### Modified VGG Convolutional deep learning network
######################################################################
# Description      Size        Name
########################################
## Sublayer 1
# INPUT:        [256x256x1]     x_
# CONV3-64:     [256x256x64]    conv11
# CONV3-64:     [256x256x64]    conv12
# POOL2:        [128x128x64]    pool1
########################################
## Sublayer 2
# CONV3-128:    [128x1128x128]  conv21
# CONV3-128:    [128x128x128]   conv22
# POOL2:        [64x64x128]     pool2
########################################
## Sublayer 3
# CONV3-256:    [64x64x256]     conv31
# CONV3-256:    [64x64x256]     conv32
# CONV3-256:    [64x64x256]     conv33
# POOL2:        [32x32x256]     pool3
########################################
## Sublayer 4
# CONV3-512:    [32x32x512]     conv41
# CONV3-512:    [32x32x512]     conv42
# CONV3-512:    [32x32x512]     conv43
# POOL2:        [16x16x512]     pool4
########################################
## Sublayer 5
# CONV3-512:    [16x16x512]     conv51
# CONV3-512:    [16x16x512]     conv52
# CONV3-512:    [16x16x512]     conv53
# POOL2:        [8x8x512]       pool5
########################################
## Fully connected layer
# FC:           [1x1x4096]      fc1
# FC:           [1x1x4096]      fc2
# FC:           [1x1x38]        fc3 (to output)
########################################

# Force CPU only mode
with tf.device('/cpu:0'):
   # Creates a session with log_device_placement set to True.
   # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
   sess = tf.Session()

   # Constants
   nClasses = 38
   imageSize = 256*256
   batchSize = 10

   # The size of the images is 256x256`
   x = tf.placeholder("float", shape=[None, imageSize], name="Input")
   # There are 4 classes (labels)
   y_ = tf.placeholder("float", shape=[None, nClasses], name = "Output")

   # FIRST SUBLAYER (64 features, 2 convolutions)
   w_conv11 = weight_variable([3, 3, 1, 32], name="Weights_conv11")
   b_conv11 = bias_variable([32], name="b_conv11")
   w_conv12 = weight_variable([3, 3, 32, 32], name="Weights_conv12")
   b_conv12 = bias_variable([32], name="b_conv12")

   # To apply the layer, we first reshape x to a 4d tensor, with the second
   # and third dimensions corresponding to image width and height,
   # and the final dimension corresponding to the number of color channels.
   x_image = tf.reshape(x, [-1,256,256,1])

   # We then convolve x_image with the weight tensor,
   # add the bias, apply the ReLU function (repeat from step 1) and finally max pool.
   h_conv11 = tf.nn.relu(conv2d(x_image, w_conv11) + b_conv11)
   h_conv12 = tf.nn.relu(conv2d(h_conv11, w_conv12) + b_conv12)
   h_pool1 = max_pool_2x2(h_conv12, name="pool1")

   # SECOND SUBLAYER (128 features, 2 convolutions)
   w_conv21 = weight_variable([3, 3, 32, 64], name="Weights_conv21")
   b_conv21 = bias_variable([64], name="b_conv21")
   w_conv22 = weight_variable([3, 3, 64, 64], name="Weights_conv22")
   b_conv22 = bias_variable([64], name="b_conv22")

   h_conv21 = tf.nn.relu(conv2d(h_pool1, w_conv21)  + b_conv21)
   h_conv22 = tf.nn.relu(conv2d(h_conv21, w_conv22) + b_conv22)
   h_pool2 = max_pool_2x2(h_conv22, name="pool2")

   # THIRD SUBLAYER (256 features, 3 convolutions)
   w_conv31 = weight_variable([3, 3, 64, 128], name="Weights_conv31")
   b_conv31 = bias_variable([128], name="b_conv31")
   w_conv32 = weight_variable([3, 3, 128, 128], name="Weights_conv32")
   b_conv32 = bias_variable([128], name="b_conv32")
   w_conv33 = weight_variable([3, 3, 128, 128], name="Weights_conv33")
   b_conv33 = bias_variable([128], name="b_conv33")

   h_conv31 = tf.nn.relu(conv2d(h_pool2, w_conv31)  + b_conv31)
   h_conv32 = tf.nn.relu(conv2d(h_conv31, w_conv32) + b_conv32)
   h_conv33 = tf.nn.relu(conv2d(h_conv32, w_conv33) + b_conv33)
   h_pool3 = max_pool_2x2(h_conv33, name="pool3")

   # FOURTH SUBLAYER (512 features, 3 convolutions)
   w_conv41 = weight_variable([3, 3, 128, 256], name="Weights_conv41")
   b_conv41 = bias_variable([256], name="b_conv41")
   w_conv42 = weight_variable([3, 3, 256, 256], name="Weights_conv42")
   b_conv42 = bias_variable([256], name="b_conv42")
   w_conv43 = weight_variable([3, 3, 256, 256], name="Weights_conv43")
   b_conv43 = bias_variable([256], name="b_conv43")

   h_conv41 = tf.nn.relu(conv2d(h_pool3, w_conv41)  + b_conv41)
   h_conv42 = tf.nn.relu(conv2d(h_conv41, w_conv42) + b_conv42)
   h_conv43 = tf.nn.relu(conv2d(h_conv42, w_conv43) + b_conv43)
   h_pool4 = max_pool_2x2(h_conv43, name="pool4")


   # DENSELY CONNECTED LAYER
   # Now that the image size has been reduced to 8x8,
   # we add a fully-connected layer with 4096 neurons to allow processing on the entire image.
   # We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight
   # matrix, add a bias, and apply a ReLU.

   ## Fully connected layers
   # FC:           [1x1x4096]      fc1 (with dropout)
   # FC:           [1x1x4096]      fc2 (with dropout)
   # FC:           [1x1x38]        fc3 (to output)

   # Fully connected layer 1 (4096 neurons)
   w_fc1 = weight_variable([16*16*256, 2048], name="Weights_fc1")
   b_fc1 = bias_variable([2048], name="biases_fc1")

   h_pool5_flat = tf.reshape(h_pool4, [-1, 16*16*256])
   h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, w_fc1) + b_fc1)
   # Dropout of fc1 (dropout keep probability of 0.5)
   keep_prob = tf.placeholder("float")
   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

   # Fully connected layer 2
   w_fc2 = weight_variable([2048, 2048], name="Weights_fc2")
   b_fc2 = bias_variable([2048], name="biases_fc2")
   h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
   h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

   # Readout layer
   w_fc3 = weight_variable([2048, nClasses], name="Weights_fc3")
   b_fc3 = bias_variable([nClasses], name="biases_fc3")
   y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)

   # Load the dataset
   datasets = dataset.read_data_sets('train', 'validation', 'whales.csv')

   # Train and eval the model
   cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
   tf.scalar_summary('cross entropy', cross_entropy)

   # train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
   train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
   correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
   sess.run(tf.initialize_all_variables())

   summary_op = tf.merge_all_summaries()
   summary_writer = tf.train.SummaryWriter('/tmp/whales', graph_def=sess.graph_def)
   saver = tf.train.Saver()

   for i in xrange(10000):
      step_start = time.time()

      batch = datasets.train.get_sequential_batch(batchSize)
      labels = batch[1]

      # Format the Y for this step
      yTrain = np.zeros((batchSize, nClasses))
      for j in xrange(batchSize):
         yTrain[j-1][ int(labels[j-1]) ] = 1
      train_step.run(feed_dict={x: batch[0], y_: yTrain, keep_prob:0.5}, session=sess)

      if i%5 == 0 and i != 0:
         #evaluate accuracy on random batch of 100 samples
         batch = datasets.train.get_random_batch(100)
         labels = batch[1]

         yTrain = np.zeros((len(batch[1]), nClasses))
         for j in xrange(len(batch[1])):
            yTrain[j][ int(labels[j]) ] = 1
         f1.write("step %d finished, time = %s\n" %(i, time.time() - step_start))
         acc, cross_entropyD, summary_str = sess.run([accuracy, cross_entropy, summary_op],
                                                   feed_dict={x: batch[0], y_: yTrain, keep_prob: 1})
         f1.write("Cross entropy = " + str(cross_entropyD) + "\n")
         f1.write("Accuracy = " + str(acc) + "\n")
         f1.write("\n--- %s seconds ---\n\n" % (time.time() - start_time))
         f1.flush()
         summary_writer.add_summary(summary_str, i)

      f1.write("\nstep %d finished, %d seconds \n" % (i, time.time() - step_start))
      f1.flush()

   # Evaluate the prediction
   test = datasets.validation.getAll()
   testLabels = test[1]

   yTest = np.zeros((len(test[1]), nClasses))
   for j in xrange(len(test[1])):
      yTest[j][ int(testLabels[j]) ] = 1

   acc, y_convD, correct_predictionD = sess.run([accuracy, y_conv, correct_prediction],
                                                feed_dict={x: test[0], y_: yTest, keep_prob: 1.0})
   f1.write("Accuracy = " + str(acc) + "\n")
   f1.write("Sum of 1: %d\n" % (sum(test[1])))
   f1.write("Correct prediction %d\n" % (sum(correct_predictionD)))
   f1.write("y %s\n" % str(test[1]))
   f1.write("y from net %s\n" % str(np.argmax(y_convD, axis=1)))
   f1.write("\n--- %s seconds ---\n\n" % (time.time() - start_time))
   f1.flush()
f1.close()
