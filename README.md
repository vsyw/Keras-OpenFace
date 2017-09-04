# Keras-OpenFace

Keras-OpenFace is a project converting [OpenFace](https://github.com/cmusatyalab/openface) from Torch implementation to a Keras version 

### If you are only interested in using pre-trained model
Load the Keras OpenFace model without local response normalization layer (Accuracy: 0.938+-0.013)
```python
from keras.models import load_model
model = load_model('./model/nn4.small2.v1.h5')
```
or load the Keras OpenFace model with local response normalization layer (Accuracy: 0.940+-0.013)
```python
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
with CustomObjectScope({'tf': tf}):
  model = load_model('./model/nn4.small2.v2.h5')
```
### Running the whole convertion process and look into Kears-Openface-convertion.ipynb
```
$ jupyter notebook
```