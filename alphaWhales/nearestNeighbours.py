import numpy as np
import dataset

class NearestNeighbor:
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the squared L2 distance
      distances = np.sum(np.square(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred


datasets = dataset.read_data_sets('train', 'validation', 'whales.csv')
train = datasets.train.getAll()
Xtr_rows = train[0]
Ytr = train[1]

validation = datasets.validation.getAll()
Xte_rows = validation[0]

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels

Yte_predict = nn.predict(Xte_rows) # predict labels on the test images

correct = sum(Yte_predict == validation[1])

# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % (float(correct)/len(validation[1]))

