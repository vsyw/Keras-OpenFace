import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Activation, concatenate
from keras.layers.normalization import BatchNormalization

mid_input = Input(shape=(192, 12, 12))
inception_3a_1x1 = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3a_1x1_conv')(mid_input)
inception_3a_1x1 = BatchNormalization(axis=1, epsilon=0.00001)(inception_3a_1x1)
inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

inception_3a_1x1_output = concatenate([inception_3a_1x1, inception_3a_1x1], axis=1)


inception_3a_1x1 = Sequential()
inception_3a_1x1.add(Conv2D(64, (1, 1), data_format='channels_first', name='inception_3a_1x1_conv', input_shape=(192, 12, 12)))
inception_3a_1x1.add(BatchNormalization(axis=1, epsilon=0.00001))
inception_3a_1x1.add(Activation('relu'))

# inception_3a_1x1_output = concatenate([inception_3a_1x1, inception_3a_1x1], axis=1)

inception_3a_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first', name='inception_3a_pool_pool')(mid_output)
inception_3a_pool = Conv2D(32, (1, 1), data_format='channels_first', name='inception_3a_pool_conv')(inception_3a_pool)
inception_3a_pool = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_pool_batchnorm')(inception_3a_pool)
inception_3a_pool = Activation('relu')(inception_3a_pool)

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

inception_3a_1x1_conv_w_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_1x1_conv_w.csv'
inception_3a_1x1_conv_b_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_1x1_conv_b.csv'
inception_3a_1x1_batchnorm_w_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_1x1_batchnorm_w.csv'
inception_3a_1x1_batchnorm_b_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_1x1_batchnorm_b.csv'
inception_3a_1x1_batchnorm_m_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_1x1_batchnorm_m.csv'
inception_3a_1x1_batchnorm_v_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_1x1_batchnorm_v.csv'

inception_3a_pool_conv_w_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_pool_conv_w.csv'
inception_3a_pool_conv_b_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_pool_conv_b.csv'
inception_3a_pool_batchnorm_w_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_pool_batchnorm_w.csv'
inception_3a_pool_batchnorm_b_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_pool_batchnorm_b.csv'
inception_3a_pool_batchnorm_m_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_pool_batchnorm_m.csv'
inception_3a_pool_batchnorm_v_path = '/Users/victor_sy_wang/Developer/ML/openface/models/openface/weights/inception_3a_pool_batchnorm_v.csv'