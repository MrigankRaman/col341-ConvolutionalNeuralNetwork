import numpy as np
from keras import layers
from keras.layers import Input,Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.utils import plot_model
import pandas as pd
from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras import initializers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dropout
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import sys
def train(traind,testd,output):
    train = pd.read_csv(traind,delimiter = ' ',header=None)
    test = pd.read_csv(testd,delimiter = ' ',header=None)
    X_train = train.values[:,:].copy()
    X_train = X_train[:,0:X_train.shape[1]-1].copy()
    Y_train = train.values[:,X_train.shape[1]].copy()
    X_test = test.values[:,:].copy()
    X_test = X_test[:,0:X_test.shape[1]-1].copy()
    X_train = X_train.astype(np.uint8)
    Y_train = Y_train.astype(np.uint8)
    X_test = X_test.astype(np.uint8)
    Y_ohe = np.zeros((Y_train.shape[0],10))
    for i in range(Y_train.shape[0]):
        Y_ohe[i,Y_train[i]] = 1
    Train_arr = np.zeros((X_train.shape[0],32,32,3))
    Test_arr =  np.zeros((X_test.shape[0],32,32,3))
    for i in range(X_train.shape[0]):
        Train_arr[i,:,:,:] = X_train[i,:].reshape(3,32,32).transpose(1,2,0)

    for i in range(X_test.shape[0]):
        Test_arr[i,:,:,:] =  X_test[i,:].reshape(3,32,32).transpose(1,2,0)
    Train_arr = Train_arr.astype(np.uint8)
    Test_arr = Test_arr.astype(np.uint8)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3,activation='relu',padding = 'same', input_shape=(32,32,3)))
    model.add(Conv2D(64, kernel_size=3,activation='relu',padding = 'same',input_shape=(32,32,3)))

    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    model.add(Conv2D(256, kernel_size=3,activation = 'relu',padding = 'same' ))
    model.add(Conv2D(256, kernel_size=3,activation = 'relu',padding = 'same' ))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))


    model.add(Conv2D(512, kernel_size=3,activation = 'relu',padding = 'same'))
    model.add(Conv2D(512, kernel_size=3,activation = 'relu',padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(512,activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    X_tr = Train_arr[0:50000,:]
    X_te = Train_arr[40000:50000,:]
    Y_tr = Y_ohe[0:50000,:]
    Y_te = Y_ohe[40000:50000,:]
    datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,vertical_flip = False,validation_split=0.1)
    datagen.fit(X_tr)
    model.fit_generator(datagen.flow(X_tr, Y_tr, batch_size=500),
                    steps_per_epoch=90, epochs=20,validation_data = (X_te,Y_te) )
    model.fit(X_tr, Y_tr, batch_size = 500,validation_data=(X_te, Y_te), epochs=5)
    model.fit_generator(datagen.flow(X_tr, Y_tr, batch_size=500),
                    steps_per_epoch=90, epochs=10,validation_data = (X_te,Y_te) )
    model.fit(X_tr, Y_tr, batch_size = 500,validation_data=(X_te, Y_te), epochs=5)
    model.fit_generator(datagen.flow(X_tr, Y_tr, batch_size=500),
                    steps_per_epoch=90, epochs=7,validation_data = (X_te,Y_te) )
    model.fit(X_tr, Y_tr, batch_size = 500,validation_data=(X_te, Y_te), epochs=5)
    model.fit_generator(datagen.flow(X_tr, Y_tr, batch_size=500),
                    steps_per_epoch=90, epochs=7,validation_data = (X_te,Y_te) )
    model.fit(X_tr, Y_tr, batch_size = 500,validation_data=(X_te, Y_te), epochs=5)
    model.fit_generator(datagen.flow(X_tr, Y_tr, batch_size=500),
                    steps_per_epoch=90, epochs=3,validation_data = (X_te,Y_te) )
    model.fit(X_tr, Y_tr, batch_size = 500,validation_data=(X_te, Y_te), epochs=2)
    
    y = model.predict(Test_arr)
    a = np.argmax(y,axis=1)
    for param in a:
        print(np.asscalar(param),file=open(output, "a"))
        print(np.asscalar(param))
if __name__ == '__main__':
    train(*sys.argv[1:])