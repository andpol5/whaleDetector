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

   def draw(self, pool, w, h, num_rows, num_cols, prefix = "filtered"):
      imgs = ['train/train/w_3.jpg', 'train/train/w_38.jpg', 'train/train/w_60.jpg']
      imgnums = [3,38,60]
      b = self.datasets.train.read_images(imgs)
      # b = np.ones((3,256,256,1))
      labels=[1,2,3]
      yTrain = np.zeros((3, self.nClasses))
      for j in xrange(3):
          yTrain[j][ labels[j] - 1 ] = 1

      img = self.sess.run([pool],feed_dict={self.x: b, self.y_: yTrain, self.keep_prob: 1})

      im = np.zeros((w*num_cols,h*num_rows))
      for k in xrange(3):
         for i in xrange(num_cols):
            for j in xrange(num_rows):
               im[w*i:w*(i+1),h*j:h*(j+1)] = np.reshape(img[0][k,:,:,j*num_cols+i], (w,h))
         im = (im - np.min(im))/(np.max(im) - np.min(im))
         cv2.imwrite("%s%d.jpg"%(prefix, imgnums[k]),im*255)


   def saliencyMap(self, var):
      whale = 'w_6527'
      imName = 'train/train/%s.jpg' % whale
      im = self.datasets.train.read_images([imName])
      yTrain = np.zeros((1, self.nClasses))
      yTrain[0][0] = 1
      cl = 13
      k = 32
      size = 8
      res = np.zeros((k,k))

      y_conv = self.sess.run([var], feed_dict = {self.x:np.array(im), self.y_: yTrain, self.keep_prob: 1})
      baseline = y_conv[0][0][cl]

      yTrain = np.zeros((k, self.nClasses))
      yTrain[:][0] = 1

      imgs = np.zeros((k*k,256,256,1))

      for i in xrange(k):
         for j in xrange(k):
            imgs[i*k+j] = im
            imgs[i*k+j,i*size:(i+1)*size,j*size:(j+1)*size] = 0
         y_conv = self.sess.run([var], feed_dict = {self.x:imgs[i*k:(i+1)*k], self.y_: yTrain, self.keep_prob: 1})
         y_conv = np.array(y_conv).reshape((k,38))
         diff = baseline - y_conv[:, cl]
         res[i, :] = diff
         print(i)

      res = ((res - np.min(res))/(np.max(res)-np.min(res))) * 255
      cv2.imwrite('saliencyMap/%s/salience_map.jpg' % whale, res)

      im = cv2.imread(imName)

      salImNew = np.zeros((im.shape[0], im.shape[1]))

      for i in xrange(salImNew.shape[0]/size):
          for j in xrange(salImNew.shape[1]/size):
              salImNew[i*size:(i+1)*size, j*size:(j+1)*size] = np.ones((size, size))*res[i,j]

      imMerged = cv2.applyColorMap(np.uint8(salImNew), cv2.COLORMAP_JET)

      imMerged = self.normalize(im+(imMerged))
      cv2.imwrite('saliencyMap/%s/merged.jpg'%whale, imMerged)

   def normalize(self, x):
      return (x-np.min(x))/(np.max(x)-np.min(x))*255





