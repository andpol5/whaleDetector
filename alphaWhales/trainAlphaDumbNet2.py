import time

import tensorflow as tf
import numpy as np

import dataset

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Constants
learningRate = 1e-4
batchSize = 20
nClasses = 38
imW = 256
imH = 256
d1 = 32
d2 = 64
d3 = 128
d4 = 256
d5 = 512
fc1 = 4096

# Force CPU only mode
with tf.device('/cpu:0'):
    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None,imW,imH,1])
    y_ = tf.placeholder("float", shape=[None, nClasses])

    # First convolution layer with pooling
    W_conv1 = weight_variable([5, 5, 1, d1])
    b_conv1 = bias_variable([d1])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # size 128*128*d1


    # Second convolution layer with pooling
    W_conv2 = weight_variable([5, 5, d1, d2])
    b_conv2 = bias_variable([d2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # size 64*64*d2


    # Third convolution layer
    W_conv3 = weight_variable([5, 5, d2, d3])
    b_conv3 = bias_variable([d3])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3) # size 32*32*d3

    # Forth convolution layer
    W_conv4 = weight_variable([5, 5, d3, d4])
    b_conv4 = bias_variable([d4])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4) # size 16*16*d4

    # Fifth convolution layer
    W_conv5 = weight_variable([5, 5, d4, d5])
    b_conv5 = bias_variable([d5])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5) # size 8*8*d5

    # Fully connected layer
    W_fc1 = weight_variable([8 * 8 * d5, fc1])
    b_fc1 = bias_variable([fc1])

    h_pool5_flat = tf.reshape(h_pool5, [-1, 8*8*d5])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    # Output layer
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([fc1, nClasses])
    b_fc2 = bias_variable([nClasses])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Cost function
    cross_entropy =  -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
    # Optimizer
    train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    datasets = dataset.read_data_sets('train/train', 'train/validation', 'whales.csv', 'whales.csv')

    sess.run(tf.initialize_all_variables())

    # validation dateset
    validation = datasets.validation.getAll()

    for i in range(20000):
        stepStart = time.time()

        batch = datasets.train.get_sequential_batch(batchSize)

        if i%25 == 0:
            train_accuracy, cross_entropyD = sess.run([accuracy, cross_entropy],
                                         feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
            print("step: %d, training accuracy: %g, time: %d\n"%(i, train_accuracy, time.time() - stepStart))
            print("step: %d, training accuracy: %g, time: %d\n"%(i, train_accuracy, time.time() - stepStart))
            print("validation accuracy: %g"%accuracy.eval(feed_dict={x:  validation[0], y_: validation[1], keep_prob: 1.0}))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('step: %d, time: %d\n' % (i, time.time() - stepStart))

    print("finale validation accuracy: %g"%accuracy.eval(feed_dict={
        x:  validation[0], y_: validation[1], keep_prob: 1.0}))
