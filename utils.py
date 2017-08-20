import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D


_FLOATX = 'float32'

def variable(value, dtype=_FLOATX, name=None):
  v = tf.Variable(np.asarray(value, dtype=dtype), name=name)
  _get_session().run(v.initializer)
  return v

def shape(x):
  return x.get_shape()

def square(x):
  return tf.square(x)

def zeros(shape, dtype=_FLOATX, name=None):
  return variable(np.zeros(shape), dtype, name)

def concatenate(tensors, axis=-1):
  if axis < 0:
      axis = axis % len(tensors[0].get_shape())
  return tf.concat(axis, tensors)

def conv2d_bn(
  x,
  cv1_name=None,
  cv1_out=None,
  cv1_filter=(1, 1),
  cv1_strides=(1, 1),
  cv2_name=None,
  cv2_out=None,
  cv2_filter=(3, 3),
  cv2_strides=(1, 1),
  padding=(1, 1),
  bn1_name=None,
  bn2_name=None
):
  tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_first', name=cv1_name)(x)
  tensor = BatchNormalization(axis=1, epsilon=0.00001, name=bn1_name)(tensor)
  tensor = Activation('relu')(tensor)
  tensor = ZeroPadding2D(padding=padding, data_format='channels_first')(tensor)
  if cv2_name == None:
    return tensor
  tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_first', name=cv2_name)(tensor)
  tensor = BatchNormalization(axis=1, epsilon=0.00001, name=bn2_name)(tensor)
  tensor = Activation('relu')(tensor)
  return tensor
