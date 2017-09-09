from keras.models import load_model
import cv2
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import utils as K

np.set_printoptions(threshold=np.nan)

path1='/Users/victor_sy_wang/Developer/ML/openface/images/examples-aligned/lennon-1.png'
path2='/Users/victor_sy_wang/Developer/ML/openface/data/lfw/dlib-affine-sz/Aaron_Eckhart/Aaron_Eckhart_0001.png'
path3='/Users/victor_sy_wang/Developer/ML/keras-facenet/data/dlib-affine-sz/Abel_Pacheco/Abel_Pacheco_0001.png'

img = cv2.imread(path2, 1)
img = img[...,::-1]
img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)

model = load_model('./model/nn4.small2.v1.h5')

x_train = np.array([img])
y = model.predict_on_batch(x_train)

print(y)
