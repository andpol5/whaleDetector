import numpy as np
import cv2

class Utility:
   def __init__(self, datasets, sess, nClasses, x, y_, keep_prob):
      self.nClasses = nClasses
      self.datasets = datasets
      self.sess = sess
      self.x = x
      self.y_ = y_
      self.keep_prob = keep_prob

   def draw(self, pool, w, h, num_rows, num_cols):
      imgs = ['batch1/w_3.jpg', 'batch1/w_38.jpg', 'batch1/w_60.jpg']
      imgnums = [3,38,60]
      b = self.datasets.train.read_images(imgs)

      labels=[1,2,3]
      yTrain = np.zeros((3, self.nClasses))
      for j in xrange(3):
          yTrain[j-1][ int(labels[j-1]) ] = 1

      img = self.sess.run([pool],feed_dict={self.x: b, self.y_: yTrain, self.keep_prob: 1})

      im = np.zeros((w*num_cols,h*num_rows))
      for k in xrange(3):
         for i in xrange(num_cols):
            for j in xrange(num_rows):
               im[w*i:w*(i+1),h*j:h*(j+1)] = np.reshape(img[0][k,:,:,j*num_cols+i], (w,h))
         cv2.imwrite("filtered%d.jpg"%imgnums[k],im)
