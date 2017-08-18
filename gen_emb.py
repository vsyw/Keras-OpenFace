from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.engine.topology import Layer
import cv2
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import utils as K

np.set_printoptions(threshold=np.nan)

w1 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l1_w.csv'
b1 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l1_b.csv'
w2 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l2_w.csv'
b2 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l2_b.csv'
m2 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l2_m.csv'
v2 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l2_v.csv'
w6 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l6_w.csv'
b6 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l6_b.csv'
w7 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l7_w.csv'
b7 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l7_b.csv'
m7 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l7_m.csv'
v7 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l7_v.csv'
w9 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l9_w.csv'
b9 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l9_b.csv'
w10 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l10_w.csv'
b10 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l10_b.csv'
m10 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l10_m.csv'
v10 = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/l10_v.csv'

l1_w = genfromtxt(w1, delimiter=',', dtype=None)
l1_w = l1_w.reshape(64, 147).reshape(64, 3, 7, 7)
l1_w = np.transpose(l1_w, (2, 3, 1, 0))
l1_b = genfromtxt(b1, delimiter=',', dtype=None)
l1 = [l1_w, l1_b]


l2_w = genfromtxt(w2, delimiter=',', dtype=None)
l2_b = genfromtxt(b2, delimiter=',', dtype=None)
l2_m = genfromtxt(m2, delimiter=',', dtype=None)
l2_v = genfromtxt(v2, delimiter=',', dtype=None)
l2 = [l2_w, l2_b, l2_m, l2_v]

l6_w = genfromtxt(w6, delimiter=',', dtype=None)
l6_w = l6_w.reshape(64, 64).reshape(64, 64, 1, 1)
l6_w = np.transpose(l6_w, (2, 3, 1, 0))
l6_b = genfromtxt(b6, delimiter=',', dtype=None)
l6 = [l6_w, l6_b]

l7_w = genfromtxt(w7, delimiter=',', dtype=None)
l7_b = genfromtxt(b7, delimiter=',', dtype=None)
l7_m = genfromtxt(m7, delimiter=',', dtype=None)
l7_v = genfromtxt(v7, delimiter=',', dtype=None)
l7 = [l7_w, l7_b, l7_m, l7_v]

l9_w = genfromtxt(w9, delimiter=',', dtype=None)
l9_w = l9_w.reshape(192, 576).reshape(192, 64, 3, 3)
l9_w = np.transpose(l9_w, (2, 3, 1, 0))
l9_b = genfromtxt(b9, delimiter=',', dtype=None)
l9 = [l9_w, l9_b]

l10_w = genfromtxt(w10, delimiter=',', dtype=None)
l10_b = genfromtxt(b10, delimiter=',', dtype=None)
l10_m = genfromtxt(m10, delimiter=',', dtype=None)
l10_v = genfromtxt(v10, delimiter=',', dtype=None)
l10 = [l10_w, l10_b, l10_m, l10_v]

class LRN2D(Layer):
  """
  This code is adapted from pylearn2.
  License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
  """

  def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    if n % 2 == 0:
      raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
    super(LRN2D, self).__init__(**kwargs)
    self.alpha = alpha
    self.k = k
    self.beta = beta
    self.n = n

  def get_output(self, train):
    X = self.get_input(train)
    b, ch, r, c = K.shape(X)
    half_n = self.n // 2
    input_sqr = K.square(X)
    extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
    input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                               input_sqr,
                               extra_channels[:, half_n + ch:, :, :]],
                              axis=1)
    scale = self.k
    for i in range(self.n):
      scale += self.alpha * input_sqr[:, i:i + ch, :, :]
    scale = scale ** self.beta
    return X / scale

  def get_config(self):
    config = {"name": self.__class__.__name__,
              "alpha": self.alpha,
              "k": self.k,
              "beta": self.beta,
              "n": self.n}
    base_config = super(LRN2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

model = Sequential()
model.add(ZeroPadding2D(padding=(3, 3), input_shape=(3, 96, 96), data_format='channels_first'))
model.add(Conv2D(64, (7, 7), strides=(2, 2), data_format='channels_first'))
model.add(BatchNormalization(axis=1, epsilon=0.00001))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=3, strides=2, data_format='channels_first'))
model.add(LRN2D())

#8 => Inception2, torch layer6
model.add(Conv2D(64, (1, 1), data_format='channels_first'))
model.add(BatchNormalization(axis=1, epsilon=0.00001))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_first'))
model.add(Conv2D(192, (3, 3), data_format='channels_first'))
model.add(BatchNormalization(axis=1, epsilon=0.00001))
model.add(Activation('relu'))
model.add(LRN2D())
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=3, strides=2, data_format='channels_first'))


model.layers[1].set_weights(l1)
model.layers[2].set_weights(l2)
model.layers[7].set_weights(l6)
model.layers[8].set_weights(l7)
model.layers[11].set_weights(l9)
model.layers[12].set_weights(l10)

img = cv2.imread('/Users/victor_sy_wang/Developer/ML/openface/data/lfw/dlib-affine-sz/Aaron_Eckhart/Aaron_Eckhart_0001.png', 1)
img = img[...,::-1]
img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
# img = np.transpose(img, (2,0,1))/255.0
# print(img[0][0][0])
# print(img[1])

x_train = np.array([img])
y = model.predict_on_batch(x_train)

print(y)
