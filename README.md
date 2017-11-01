# Keras-OpenFace

Keras-OpenFace is a project converting [OpenFace](https://github.com/cmusatyalab/openface) from it's original Torch implementation to a Keras version 

### If you are only interested in using pre-trained model
Load the Keras OpenFace model(Accuracy: 0.938+-0.013)
```python
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
with CustomObjectScope({'tf': tf}):
  model = load_model('./model/nn4.small2.v1.h5')
```
### Running the whole convertion process and look into Kears-Openface-convertion.ipynb
```
$ jupyter notebook
```

### CoreML-OpenFace
Pre-trained CoreML version of OpenFace in model/openface.coreml which you can easily integrate OpenFace into your iOS application.