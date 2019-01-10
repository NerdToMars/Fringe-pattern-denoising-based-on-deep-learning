# Fringe-pattern-denoising-based-on-deep-learning
Keras model, please change to your data or your perfered model

## Training
make sure your data folder structure is correct:

Tran_data
- data_[number]
	-1.bmp
	- 2.bmp
	- ...
	- ground.bmp #ground truth

Val_data
- data_[number]
	-1.bmp
	- 2.bmp
	- ...
	- ground.bmp #ground truth

1. modify the steps and epochs in main.py
2. run the main.py

or you can use jupyter

================================================================================================

# LSTM UNET TEST


```python
from model import *
from data import *
import os
import sys

ROOTPATH = os.path.abspath("./")
sys.path.append(ROOTPATH)
train_dir = "./TRAIN/"
val_dir = "./VAL/" 

test_dir = "./test/"

all_list = os.listdir(train_dir)
train_dataset = [train_dir+s_dir+'/' for s_dir in all_list]

all_list = os.listdir(val_dir)
val_dataset = [val_dir+s_dir+'/' for s_dir in all_list]




```

    /usr/local/lib/python3.5/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


# Create data generator


```python
train_data_generator = data_generator(train_dataset,batch_size=1)
val_data_generator = data_generator(val_dataset,batch_size=1)
```

# Tensorboard records (optional) 
need install tensorboard


```python
from keras.callbacks import TensorBoard
class TB(TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0
    
    def on_batch_end(self, batch, logs=None):
        self.counter+=1
        if self.counter%self.log_every==0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()
        
        super().on_batch_end(batch, logs)
        
tensorboard_log = TB(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
print(model.summary())
```

# Compile model


```python
model = LSTMUnet()
model_checkpoint = ModelCheckpoint('lstm_unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
print(model.summary())

```

# Train


```python
model.fit_generator(
    train_data_generator,
    validation_data=val_data_generator,
    validation_steps =400,
    steps_per_epoch=2325,
    epochs=15,
    callbacks=[model_checkpoint, tensorboard_log])

```

# Testing


```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```


```python
model_val = LSTMUnet()
model_val.load_weights('lstm_3.hdf5') # load your pretrained model
```


```python
test_dir = "/cole_driver/dff/PLS3Bx10/"
test_dir2 = "/cole_driver/dff/PLS3Sx50/" #change to your test folder

input_ = data_feed(test_dir)
input_s = np.zeros((1,)+input_.shape,dtype=np.uint8)
input_s[0] = input_

out_d = model_val.predict(input_s)
imd = plt.imshow(out_d[0,::,::,0])
plt.colorbar()
```


================================================================================================


For article, please cite：Fringe pattern denoising based on deep learning   DOI: 10.1016/j.optcom.2018.12.058

Yan, Ketao, Yingjie Yu, Chongtian Huang, Liansheng Sui, Kemao Qian, and Anand Asundi. "Fringe pattern denoising based on deep learning." Optics Communications (2018).

@article{yan2018fringe,
  title={Fringe pattern denoising based on deep learning},
  author={Yan, Ketao and Yu, Yingjie and Huang, Chongtian and Sui, Liansheng and Qian, Kemao and Asundi, Anand},
  journal={Optics Communications},
  year={2018},
  publisher={Elsevier}
}
