import time
import signal
import sys

import tensorflow as tf
import numpy as np

import dataset

np.set_printoptions(threshold=np.nan)
f1 = open('log_ds_%d' % (time.time()), 'w+')

def log(str):
    f1.write(str)
    f1.flush()
    print(str)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=0.001, stddev=0.3)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def activation(x):
    return tf.nn.relu(x)

def normalize(x):
    return tf.nn.local_response_normalization(x)

# Constants
starter_learning_rate = 1e-4
batchSize = 20
# dropout = 0.8

nClasses = 38
imW = 256
imH = 256
fc1 = 1024

# Force CPU only mode
with tf.device('/cpu:0'):
    # create session
    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None,imW,imH,1])
    y_ = tf.placeholder("float", shape=[None, nClasses])

    # First sub layer
    # 3x3, 32
    # pooling
    # First convolution layer with pooling
    w_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = activation(conv2d(x, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1) # size 128*128*32

    # Second sub layer
    # 3x3, 64
    # pooling
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = activation(conv2d(normalize(h_pool1), w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # size 64*64*64

    # Third sublayer layer
    # 3x3, 64
    # 3x3, 128
    # 3x3, 128
    # pooling
    w_conv31 = weight_variable([3, 3, 64, 64])
    b_conv31 = bias_variable([64])
    h_conv31 = activation(conv2d(normalize(h_pool2), w_conv31) + b_conv31)

    w_conv32 = weight_variable([3, 3, 64, 128])
    b_conv32 = bias_variable([128])
    h_conv32 = activation(conv2d(normalize(h_conv31), w_conv32) + b_conv32)

    w_conv33 = weight_variable([3, 3, 128, 128])
    b_conv33 = bias_variable([128])
    h_conv33 = activation(conv2d(normalize(h_conv32), w_conv33) + b_conv33)

    h_pool3 = max_pool_2x2(h_conv33) # size 32*32*128

    # Fourth sub layer
    # 3x3, 256
    # 3x3, 256
    # pooling
    w_conv41 = weight_variable([3, 3, 128, 256])
    b_conv41 = bias_variable([256])
    h_conv41 = activation(conv2d(normalize(h_pool3), w_conv41) + b_conv41)

    w_conv42 = weight_variable([3, 3, 256, 256])
    b_conv42 = bias_variable([256])
    h_conv42 = activation(conv2d(normalize(h_conv41), w_conv42) + b_conv42)

    h_pool4 = max_pool_2x2(h_conv42) # size 16*16*256

    # Fifth sub layer
    # 3x3, 256
    # 3x3, 256
    # pooling
    w_conv51 = weight_variable([3, 3, 256, 256])
    b_conv51 = bias_variable([256])
    h_conv51 = activation(conv2d(normalize(h_pool4), w_conv51) + b_conv51)

    w_conv52 = weight_variable([3, 3, 256, 256])
    b_conv52 = bias_variable([256])
    h_conv52 = activation(conv2d(normalize(h_conv51), w_conv52) + b_conv52)

    h_pool5 = max_pool_2x2(h_conv52) # size 8*8*256

    # Sixth sub layer
    # 3x3, 256
    # 3x3, 256
    # pooling
    w_conv61 = weight_variable([3, 3, 256, 256])
    b_conv61 = bias_variable([256])
    h_conv61 = activation(conv2d(normalize(h_pool5), w_conv61) + b_conv61)

    w_conv62 = weight_variable([3, 3, 256, 256])
    b_conv62 = bias_variable([256])
    h_conv62 = activation(conv2d(normalize(h_conv61), w_conv62) + b_conv62)

    h_pool6 = max_pool_2x2(h_conv62) # size 4*4*256

    # Fully connected layer
    w_fc1 = weight_variable([4*4*256, 1024])
    b_fc1 = bias_variable([1024])

    h_pool6_flat = tf.reshape(normalize(h_pool6), [-1, 4*4*256])
    h_fc1 = activation(tf.matmul(h_pool6_flat, w_fc1) + b_fc1)

    # Output layer
    keep_prob = tf.placeholder("float")
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    w_fc2 = weight_variable([1024, nClasses])
    b_fc2 = bias_variable([nClasses])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

    # Cost function
    cross_entropy =  -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-15, 1.0)))

    # Exponentially decaying learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.98, staircase=True)
    # Passing global_step to minimize() will increment it at each step.

    # Optimizer
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    datasets = dataset.read_data_sets('train/train', 'train/validation', 'whales.csv', 'whales.csv')

    sess.run(tf.initialize_all_variables())

    # weights control
    # W_conv1_mean = tf.reduce_mean(tf.reduce_mean(tf.abs(W_conv1), 1), 0)
    # W_conv2_mean = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(W_conv2), 0), 0), 0)
    # W_conv3_mean = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(W_conv3), 0), 0), 0)
    # W_conv4_mean = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(W_conv4), 0), 0), 0)
    # W_conv5_mean = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(W_conv5), 0), 0), 0)
    # W_fc1_mean = tf.reduce_mean(tf.abs(W_fc1), 0)
    w_fc2_mean = tf.reduce_mean(tf.abs(w_fc2))

    h_conv1_mean = tf.reduce_mean(tf.abs(h_conv1))
    h_conv2_mean = tf.reduce_mean(tf.abs(h_conv2))
    h_conv3_mean = tf.reduce_mean(tf.abs(h_conv33))
    h_conv4_mean = tf.reduce_mean(tf.abs(h_conv42))
    h_conv5_mean = tf.reduce_mean(tf.abs(h_conv52))
    h_fc1_mean = tf.reduce_mean(tf.abs(h_fc1))

    # validation dateset
    validation = datasets.validation.getAll()
    # entireTrainSet = datasets.train.getAll()

    saver = tf.train.Saver()

    # Restore previous model here if applicable
    previous_model_name = 'deep_sense-model-15-percent'
    saver.restore(sess, previous_model_name)

    for i in range(20000):
        stepStart = time.time()

        batch = datasets.train.get_sequential_batch(batchSize)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        if i%25 == 0:
            train_accuracy, cross_entropyD, \
            h_conv1_meanD, h_conv2_meanD, h_conv3_meanD, h_conv4_meanD, h_conv5_meanD,\
            h_fc1_meanD, W_fc2_meanD,\
            yD, currentLearningRate \
            = sess.run([accuracy, cross_entropy,
                        h_conv1_mean, h_conv2_mean, h_conv3_mean, h_conv4_mean, h_conv5_mean,
                        h_fc1_mean, w_fc2_mean, y_conv, learning_rate],
                        feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            log("step: %d, training accuracy: %f, time: %d"%(i, train_accuracy, time.time() - stepStart))
            log("train cross entropy: %f"%(cross_entropyD))
            log("h_conv1 mean = %s"%(h_conv1_meanD))
            log("h_conv2 mean = %s"%(h_conv2_meanD))
            log("h_conv3 mean = %s"%(h_conv3_meanD))
            log("h_conv4 mean = %s"%(h_conv4_meanD))
            log("h_conv5 mean = %s"%(h_conv5_meanD))
            log("h_fc1 mean = %s"%(h_fc1_meanD))
            log("w_fc2 mean = %s"%(W_fc2_meanD))
            log(" learning rate = %f" % (currentLearningRate));
            log("y     = %s"%(str(np.argmax(yD, axis=1))))
            log("yReal = %s"%(str(np.argmax(batch[1], axis=1))))
            log("validation accuracy: %g"%accuracy.eval(feed_dict={x:  validation[0], y_: validation[1], keep_prob: 1.0}))

        if i%500 == 0 and i != 0:
            saver.save(sess, 'deep_sense-model-%d' % (time.time()), global_step=global_step)

        log('step: %d, time: %d\n' % (i, time.time() - stepStart))

    log("final validation accuracy: %g"%accuracy.eval(feed_dict={
        x:  validation[0], y_: validation[1], keep_prob: 1.0}))
