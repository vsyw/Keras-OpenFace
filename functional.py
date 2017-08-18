from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Activation, concatenate
from keras.layers.normalization import BatchNormalization

img_input = Input(shape=(3, 96, 96))
