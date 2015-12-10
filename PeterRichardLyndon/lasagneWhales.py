#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time
import re
from collections import defaultdict

import numpy as np
from scipy.ndimage import imread
import theano
import theano.tensor as T

import lasagne


##############################
## DATASET GLOBALS
IM_WIDTH  = 128
IM_HEIGHT = 128
NUM_CLASSES = 3
BATCH_SIZE = 15
##############################


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
   # Load all the images
   inputDir = 'imgs'
   labelsFile = 'whales.csv'
   output = np.genfromtxt(labelsFile, skip_header=1, dtype=[('image', 'S10'), ('label', 'S11')], delimiter=',')
   labels = [x[1] for x in output]
   numberOfClasses = len(set(labels))
   
   # Read the images into 4D array (numImages, numChannels, width, height)
   # Assume all files in dir are images and all are the same size
   fileNames = os.listdir(inputDir)
   im = imread(os.path.join(inputDir, fileNames[0]))
   w, h = im.shape
   
   # Initialize the ndarray
   images = np.ndarray((len(fileNames), 1, w, h))
   # Load the first image into it's slot
   images[0,0,:,:] = im
   
   for i in xrange(1, len(fileNames)):
      im = imread(os.path.join(inputDir, fileNames[i]))
      images[i,0:,:] = im
   
   labelsDict = {int(re.search("w_(\\d+)\.jpg", x[0]).group(1)):int(re.search("whale_(\\d+)", x[1]).group(1)) for x in output}
   
   # to provide default values
   labelsDict = defaultdict(lambda:0, labelsDict)
   examples = [int(re.search("w_(\\d+).jpg",x).group(1)) for x in fileNames]
   labels = [labelsDict[x] for x in examples]
   labels = np.array(labels)
   
   origTrainLabels = labels[labels > 0]

   # Renumber the labels to have consecutive numnbers   
   l = np.zeros(max(origTrainLabels))
   i = 0
   for k in sorted(set(origTrainLabels)):
      l[k-1] = i
      i += 1
   trainLabels = np.array([l[x-1] for x in origTrainLabels])
   trainLabels = trainLabels.astype(np.uint8)
   
   # We can now download and read the training and test set images and labels.
   X_train = images
   y_train = trainLabels
   X_test = images
   y_test = trainLabels
   
   X_val = images
   y_val = trainLabels
   # We reserve the last 10000 training examples for validation.
#    X_train, X_val = X_train[:-10000], X_train[-10000:]
#    y_train, y_val = y_train[:-10000], y_train[-10000:]
   
   # We just return all the arrays in order, as expected in main().
   # (It doesn't matter how we do this as long as we can read them again.)
   return X_train, y_train, X_val, y_val, X_test, y_test

def build_cnn(input_var=None):
   # As a third model, we'll create a CNN of two convolution + pooling stages
   # and a fully-connected hidden layer in front of the output layer.
   
   # Input layer, as usual:
   network = lasagne.layers.InputLayer(shape=(None, 1, IM_HEIGHT, IM_WIDTH),
                                       input_var=input_var)
   # This time we do not apply input dropout, as it tends to work less well
   # for convolutional layers.
   
   # Convolutional layer with 32 kernels of size 5x5. Strided and padded
   # convolutions are supported as well; see the docstring.
   network = lasagne.layers.Conv2DLayer(
           network, num_filters=32, filter_size=(5, 5),
           nonlinearity=lasagne.nonlinearities.rectify,
           W=lasagne.init.GlorotUniform())
   # Expert note: Lasagne provides alternative convolutional layers that
   # override Theano's choice of which implementation to use; for details
   # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
   
   # Max-pooling layer of factor 2 in both dimensions:
   network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
   
   # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
   network = lasagne.layers.Conv2DLayer(
           network, num_filters=32, filter_size=(5, 5),
           nonlinearity=lasagne.nonlinearities.rectify)
   network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
   
   # And another convolution with 32 5x5 kernels, and another 2x2 pooling:
   network = lasagne.layers.Conv2DLayer(
           network, num_filters=32, filter_size=(5, 5),
           nonlinearity=lasagne.nonlinearities.rectify)
   network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
   
   # And another convolution with 32 5x5 kernels, and another 2x2 pooling:
   network = lasagne.layers.Conv2DLayer(
           network, num_filters=32, filter_size=(5, 5),
           nonlinearity=lasagne.nonlinearities.rectify)
   network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
      
   # A fully-connected layer of 512 units with 50% dropout on its inputs:
   network = lasagne.layers.DenseLayer(
           lasagne.layers.dropout(network, p=.5),
           num_units=512,
           nonlinearity=lasagne.nonlinearities.rectify)
   
   # And, finally, the 3-unit output layer with 50% dropout on its inputs:
   network = lasagne.layers.DenseLayer(
           lasagne.layers.dropout(network, p=.5),
           num_units=NUM_CLASSES,
           nonlinearity=lasagne.nonlinearities.softmax)
   
   return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
   assert len(inputs) == len(targets)
   if shuffle:
      indices = np.arange(len(inputs))
      np.random.shuffle(indices)
   for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
      if shuffle:
         excerpt = indices[start_idx:start_idx + batchsize]
      else:
         excerpt = slice(start_idx, start_idx + batchsize)
      yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='mlp', num_epochs=500):
   # Load the dataset
   print("Loading data...")
   X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
   
   # Prepare Theano variables for inputs and targets
   input_var = T.tensor4('inputs')
   target_var = T.ivector('targets')
   
   # Create neural network model (depending on first command line parameter)
   network = build_cnn(input_var)

   # Create a loss expression for training, i.e., a scalar objective we want
   # to minimize (for our multi-class problem, it is the cross-entropy loss):
   prediction = lasagne.layers.get_output(network)
   loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
   loss = loss.mean()
   # We could add some weight decay as well here, see lasagne.regularization.
   
   # Create update expressions for training, i.e., how to modify the
   # parameters at each training step. Here, we'll use Stochastic Gradient
   # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
   params = lasagne.layers.get_all_params(network, trainable=True)
   updates = lasagne.updates.nesterov_momentum(
           loss, params, learning_rate=0.01, momentum=0.9)
   
   # Create a loss expression for validation/testing. The crucial difference
   # here is that we do a deterministic forward pass through the network,
   # disabling dropout layers.
   test_prediction = lasagne.layers.get_output(network, deterministic=True)
   test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
   test_loss = test_loss.mean()
   # As a bonus, also create an expression for the classification accuracy:
   test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
   
   # Compile a function performing a training step on a mini-batch (by giving
   # the updates dictionary) and returning the corresponding training loss:
   train_fn = theano.function([input_var, target_var], loss, updates=updates)
   
   # Compile a second function computing the validation loss and accuracy:
   val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

   # Finally, launch the training loop.
   print("Starting training...")
   # We iterate over epochs:
   for epoch in range(num_epochs):
      # In each epoch, we do a full pass over the training data:
      train_err = 0
      train_batches = 0
      start_time = time.time()
      for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
         inputs, targets = batch
         train_err += train_fn(inputs, targets)
         train_batches += 1

         # And a full pass over the validation data:
         val_err = 0
         val_acc = 0
         val_batches = 0
         for batch in iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

      # Then we print the results for this epoch:
      print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
      print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
      print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
      print("  validation accuracy:\t\t{:.2f} %".format(
          val_acc / val_batches * 100))

   # After training, we compute and print the test error:
   test_err = 0
   test_acc = 0
   test_batches = 0
   for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
      inputs, targets = batch
      err, acc = val_fn(inputs, targets)
      test_err += err
      test_acc += acc
      test_batches += 1
   print("Final results:")
   print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
   print("  test accuracy:\t\t{:.2f} %".format(
       test_acc / test_batches * 100))

   # Optionally, you could now dump the network weights to a file like this:
   # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
   #
   # And load them again later on like this:
   # with np.load('model.npz') as f:
   #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
   # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
   if ('--help' in sys.argv) or ('-h' in sys.argv):
      print("Trains a neural network on MNIST using Lasagne.")
      print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
      print()
      print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
      print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
      print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
      print("       input dropout and DROP_HID hidden dropout,")
      print("       'cnn' for a simple Convolutional Neural Network (CNN).")
      print("EPOCHS: number of training epochs to perform (default: 500)")
   else:
      kwargs = {}
      if len(sys.argv) > 1:
         kwargs['model'] = sys.argv[1]
      if len(sys.argv) > 2:
         kwargs['num_epochs'] = int(sys.argv[2])
   main(**kwargs)
