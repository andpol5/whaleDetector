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

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, mean=0.001, stddev=0.3)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = name)

def activation(x):
    return tf.nn.relu(x)

def normalize(x):
    return tf.nn.local_response_normalization(x)

# Constants
starter_learning_rate = 1e-5
batchSize = 30
dropout = 1

nClasses = 38
imW = 256
imH = 256
fc1 = 1024

# Force CPU only mode
with tf.device('/cpu:0'):
    # create session
    sess = tf.InteractiveSession()

    log('nClasses: %d, imageSize: %d, batchSize: %d, learningRate: %e, dropOut: %f\n'
                    % (nClasses, imW, batchSize, starter_learning_rate, dropout))

    x = tf.placeholder("float", shape=[None,imW,imH,1])
    y_ = tf.placeholder("float", shape=[None, nClasses])

    # FIRST SUBLAYER (64 features, 2 convolutions)
    w_conv11 = weight_variable([3, 3, 1, 32], name="Weights_conv11")
    b_conv11 = bias_variable([32], name="b_conv11")
    w_conv12 = weight_variable([3, 3, 32, 32], name="Weights_conv12")
    b_conv12 = bias_variable([32], name="b_conv12")

    # We then convolve x_image with the weight tensor,
    # add the bias, apply the ReLU function (repeat from step 1) and finally max pool.
    h_conv11 = activation(conv2d(x, w_conv11) + b_conv11)
    h_conv12 = activation(conv2d(normalize(h_conv11), w_conv12) + b_conv12)
    h_pool1 = max_pool_2x2(h_conv12, name="pool1")

    # SECOND SUBLAYER (128 features, 2 convolutions)
    w_conv21 = weight_variable([3, 3, 32, 64], name="Weights_conv21")
    b_conv21 = bias_variable([64], name="b_conv21")
    w_conv22 = weight_variable([3, 3, 64, 64], name="Weights_conv22")
    b_conv22 = bias_variable([64], name="b_conv22")

    h_conv21 = activation(conv2d(normalize(h_pool1), w_conv21)  + b_conv21)
    h_conv22 = activation(conv2d(normalize(h_conv21), w_conv22) + b_conv22)
    h_pool2 = max_pool_2x2(h_conv22, name="pool2")

    # THIRD SUBLAYER (256 features, 3 convolutions)
    w_conv31 = weight_variable([3, 3, 64, 128], name="Weights_conv31")
    b_conv31 = bias_variable([128], name="b_conv31")
    w_conv32 = weight_variable([3, 3, 128, 128], name="Weights_conv32")
    b_conv32 = bias_variable([128], name="b_conv32")
    w_conv33 = weight_variable([3, 3, 128, 128], name="Weights_conv33")
    b_conv33 = bias_variable([128], name="b_conv33")

    h_conv31 = activation(conv2d(normalize(h_pool2), w_conv31)  + b_conv31)
    h_conv32 = activation(conv2d(normalize(h_conv31), w_conv32) + b_conv32)
    h_conv33 = activation(conv2d(normalize(h_conv32), w_conv33) + b_conv33)
    h_pool3 = max_pool_2x2(h_conv33, name="pool3")

    # FOURTH SUBLAYER (512 features, 3 convolutions)
    w_conv41 = weight_variable([3, 3, 128, 256], name="Weights_conv41")
    b_conv41 = bias_variable([256], name="b_conv41")
    w_conv42 = weight_variable([3, 3, 256, 256], name="Weights_conv42")
    b_conv42 = bias_variable([256], name="b_conv42")
    w_conv43 = weight_variable([3, 3, 256, 256], name="Weights_conv43")
    b_conv43 = bias_variable([256], name="b_conv43")

    h_conv41 = activation(conv2d(normalize(h_pool3), w_conv41)  + b_conv41)
    h_conv42 = activation(conv2d(normalize(h_conv41), w_conv42) + b_conv42)
    h_conv43 = activation(conv2d(normalize(h_conv42), w_conv43) + b_conv43)
    h_pool4 = max_pool_2x2(h_conv43, name="pool4")


    # FIFTH SUBLAYER (512 features, 3 convolutions)
    w_conv51 = weight_variable([3, 3, 256, 256], name="Weights_conv51")
    b_conv51 = bias_variable([256], name="b_conv51")
    w_conv52 = weight_variable([3, 3, 256, 256], name="Weights_conv52")
    b_conv52 = bias_variable([256], name="b_conv52")
    w_conv53 = weight_variable([3, 3, 256, 256], name="Weights_conv53")
    b_conv53 = bias_variable([256], name="b_conv53")

    h_conv51 = activation(conv2d(normalize(h_pool4), w_conv51)  + b_conv51)
    h_conv52 = activation(conv2d(normalize(h_conv51), w_conv52) + b_conv52)
    h_conv53 = activation(conv2d(normalize(h_conv52), w_conv53) + b_conv53)
    h_pool5 = max_pool_2x2(h_conv53, name="pool5")


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
    w_fc1 = weight_variable([8*8*256, 2048], name="Weights_fc1")
    b_fc1 = bias_variable([2048], name="biases_fc1")

    h_pool5_flat = tf.reshape(h_pool5, [-1, 8*8*256])
    h_fc1 = activation(tf.matmul(h_pool5_flat, w_fc1) + b_fc1)
    # Dropout of fc1 (dropout keep probability of 0.5)
    keep_prob = tf.placeholder("float")
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Fully connected layer 2
    w_fc2 = weight_variable([2048, 2048], name="Weights_fc2")
    b_fc2 = bias_variable([2048], name="biases_fc2")
    h_fc2 = activation(tf.matmul(h_fc1, w_fc2) + b_fc2)
    # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # Readout layer
    w_fc3 = weight_variable([2048, nClasses], name="Weights_fc3")
    b_fc3 = bias_variable([nClasses], name="biases_fc3")
    y_conv = tf.nn.softmax(tf.matmul(h_fc2, w_fc3) + b_fc3)
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

    # validation dateset
    validation = datasets.validation.getAll()
    # entireTrainSet = datasets.train.getAll()

    saver = tf.train.Saver()

    # Restore previous model here if applicable
    # previous_model_name = 'my-model-1453028131-8001'
    # saver.restore(sess, previous_model_name)

    for i in range(20000):
        stepStart = time.time()

        batch = datasets.train.get_sequential_batch(batchSize)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout})
        if i%25 == 0:
            train_accuracy, cross_entropyD,\
            yD, currentLearningRate \
            = sess.run([accuracy, cross_entropy, y_conv, learning_rate],
                        feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
            log("step: %d, training accuracy: %f, time: %d"%(i, train_accuracy, time.time() - stepStart))
            log("train cross entropy: %f"%(cross_entropyD))
            log("learning rate = %.10f" % (currentLearningRate));
            log("y     = %s"%(str(np.argmax(yD, axis=1))))
            log("yReal = %s"%(str(np.argmax(batch[1], axis=1))))
            # log("validation accuracy: %g"%accuracy.eval(feed_dict={x:  validation[0], y_: validation[1], keep_prob: 1.0}))

        if i%500 == 0 and i != 0:
            saver.save(sess, 'deep_sense-model-%d' % (time.time()), global_step=global_step)

        log('step: %d, time: %d\n' % (i, time.time() - stepStart))

    log("final validation accuracy: %g"%accuracy.eval(feed_dict={
        x:  validation[0], y_: validation[1], keep_prob: 1.0}))
