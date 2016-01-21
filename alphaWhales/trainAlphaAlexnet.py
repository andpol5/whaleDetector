#! /usr/bin/python
import numpy as np
import tensorflow as tf
import time
import dataset

np.set_printoptions(threshold=np.nan)
f1 = open('log_alexnet_%d' % (time.time()), 'w+')

def log(str):
    f1.write(str)
    f1.flush()
    print(str)

# Tensorflow convinience functions
def weight_variable(shape, name):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial, name=name)

def bias_variable(shape, name):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial, name=name)

def conv2d(x, W):
   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x, name):
   return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

def activation(x):
    return tf.nn.relu(x)

def normalize(x):
    return tf.nn.local_response_normalization(x)


start_time = time.time()
######################################################################
### Alex net
######################################################################
# Description      Size        Name
########################################
## Sublayer 1
# INPUT:        [227x227x1]     x_
# CONV1-96:     [55x55x96]    conv1
# POOL1:        [27x27x96]    pool1
########################################
## Sublayer 2
# CONV2-256:    [27x27x256]  conv2
# POOL2:        [13x13x256]  pool2
########################################
## Sublayer 3
# CONV3-384:    [13x13x384]     conv31
# CONV3-384:    [13x13x384]     conv32
# CONV3-256:    [13x13x256]     conv33
# POOL3:        [6x6x256]     pool3
########################################
## Fully connected layer
# FC:           [1x1x4096]      fc1
# FC:           [1x1x4096]      fc2
# FC:           [1x1x38]        fc3 (to output)
########################################
def doAlexNet(trainDir, valDir, trainCsv, valCsv):

   f1=open('log_%d' % (time.time()), 'w+')
   f1.write('AAAAA\n')
   f1.write("Start %s\n" % time.time())
   f1.flush()
   # Force CPU only mode
   with tf.device('/cpu:0'):
      # Creates a session with log_device_placement set to True.
      # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
      sess = tf.Session()

      # Constants
      nClasses = 38
      imageSize = 227
      starter_learning_rate = 1e-5
      batchSize = 10
      dropOutValue = 1
      log('nClasses: %d, imageSize: %d, batchSize: %d, learningRate: %e, dropOut: %f\n'
                    % (nClasses, imageSize, batchSize, starter_learning_rate, dropOutValue))

      # The size of the images is 227x227
      x = tf.placeholder("float", shape=[None, imageSize, imageSize, 1], name="Input")
      # There are 4 classes (labels)
      y_ = tf.placeholder("float", shape=[None, nClasses], name = "Output")

      # FIRST SUBLAYER (96 features, 1 convolution)
      w_conv1 = weight_variable([11, 11, 1, 96], name="Weights_conv1")
      b_conv1 = bias_variable([96], name="b_conv1")


      # We then convolve x_image with the weight tensor,
      # add the bias, apply the ReLU function (repeat from step 1) and finally max pool.
      h_conv1 = activation(tf.nn.conv2d(x, w_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1)
      h_pool1 = max_pool_3x3(h_conv1, name="pool1")

      # SECOND SUBLAYER (256 features, 1 convolution)
      w_conv2 = weight_variable([5, 5, 96, 256], name="Weights_conv2")
      b_conv2 = bias_variable([256], name="b_conv2")

      h_conv2 = tf.nn.relu(conv2d(normalize(h_pool1), w_conv2)  + b_conv2)
      h_pool2 = max_pool_3x3(h_conv2, name="pool2")

      # THIRD SUBLAYER (384 features, 3 convolutions)
      w_conv31 = weight_variable([3, 3, 256, 384], name="Weights_conv31")
      b_conv31 = bias_variable([384], name="b_conv31")
      w_conv32 = weight_variable([3, 3, 384, 384], name="Weights_conv32")
      b_conv32 = bias_variable([384], name="b_conv32")
      w_conv33 = weight_variable([3, 3, 384, 256], name="Weights_conv33")
      b_conv33 = bias_variable([256], name="b_conv33")

      h_conv31 = activation(conv2d(normalize(h_pool2), w_conv31)  + b_conv31)
      h_conv32 = activation(conv2d(normalize(h_conv31), w_conv32) + b_conv32)
      h_conv33 = activation(conv2d(normalize(h_conv32), w_conv33) + b_conv33)
      h_pool3 = max_pool_3x3(h_conv33, name="pool3")

      # DENSELY CONNECTED LAYER
      # Now that the image size has been reduced to 6x6,
      # we add a fully-connected layer with 4096 neurons to allow processing on the entire image.
      # We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight
      # matrix, add a bias, and apply a ReLU.

      ## Fully connected layers
      # FC:           [1x1x4096]      fc1 (with dropout)
      # FC:           [1x1x4096]      fc2 (with dropout)
      # FC:           [1x1x38]        fc3 (to output)

      # Fully connected layer 1 (4096 neurons)
      w_fc1 = weight_variable([6*6*256, 4096], name="Weights_fc1")
      b_fc1 = bias_variable([4096], name="biases_fc1")

      h_pool3_flat = tf.reshape(h_pool3, [-1, 6*6*256])
      h_fc1 = activation(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)
      # Dropout of fc1 (dropout keep probability of 0.5)
      keep_prob = tf.placeholder("float")
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      # Fully connected layer 2
      w_fc2 = weight_variable([4096, 4096], name="Weights_fc2")
      b_fc2 = bias_variable([4096], name="biases_fc2")
      h_fc2 = activation(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
      h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

      # Readout layer
      w_fc3 = weight_variable([4096, nClasses], name="Weights_fc3")
      b_fc3 = bias_variable([nClasses], name="biases_fc3")
      y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)

      # Load the dataset
      datasets = dataset.read_data_sets(trainDir, valDir, trainCsv, valCsv)

      # Train and eval the model
      cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))

      # Exponentially decaying learning rate
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.98, staircase=True)
      # Passing global_step to minimize() will increment it at each step.

      # Optimizer
      train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      sess.run(tf.initialize_all_variables())

      saver = tf.train.Saver()

      # saver.restore(sess, 'my-model-batch1-10000')

      for i in xrange(1500):
         step_start = time.time()

         batch = datasets.train.get_sequential_batch(batchSize)

         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:dropOutValue}, session=sess)

         if i%25 == 0:
            train_accuracy, cross_entropyD,\
            yD, currentLearningRate \
            = sess.run([accuracy, cross_entropy, y_conv, learning_rate],
                        feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
            log("step: %d, training accuracy: %f, time: %d"%(i, train_accuracy, time.time() - step_start))
            log("train cross entropy: %f"%(cross_entropyD))
            log("learning rate = %.10f" % (currentLearningRate));
            log("y     = %s"%(str(np.argmax(yD, axis=1))))
            log("yReal = %s"%(str(np.argmax(batch[1], axis=1))))
            # log("validation accuracy: %g"%accuracy.eval(feed_dict={x:  validation[0], y_: validation[1], keep_prob: 1.0}))

         if i%200 == 0 and i != 0:
            saver.save(sess, 'alexnet-model-%d' % (time.time()), global_step=global_step)

         log('step: %d, time: %d\n' % (i, time.time() - step_start))



doAlexNet('train2/train', 'train2/validation', 'train2/train/train.csv', 'whales.csv')