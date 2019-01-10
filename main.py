from model import *
from data import *

# the absolute folder path of your data
# suggestion-> train:val = 8:2
train_dir = "./Val_data/"
val_dir = "./Train_data/"


all_list = os.listdir(train_dir)
# absolute path of all the data
train_dataset = [train_dir+s_dir+'/' for s_dir in all_list]

all_list = os.listdir(val_dir)
val_dataset = [val_dir+s_dir+'/' for s_dir in all_list]


train_data_generator = data_generator(train_dataset,batch_size=1)
val_data_generator = data_generator(val_dataset,batch_size=1)

model = LSTMUnet()
model_checkpoint = ModelCheckpoint('cnn.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(train_data_generator,validation_data=val_data_generator,validation_steps =10,steps_per_epoch=8000,epochs=1,callbacks=[model_checkpoint])

# train_generator,
            # initial_epoch=self.epoch,
            # epochs=epochs,
            # steps_per_epoch=self.config.STEPS_PER_EPOCH,
            # callbacks=callbacks,
            # validation_data=val_generator,
            # validation_steps=self.config.VALIDATION_STEPS,
            # max_queue_size=1,
            # workers=1,
            # use_multiprocessing=False,

