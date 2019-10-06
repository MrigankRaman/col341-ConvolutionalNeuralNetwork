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
    np.random.seed(1)
    ini = initializers.glorot_uniform()
    ini1 = initializers.glorot_uniform()
    ini2 = initializers.glorot_uniform()
    ini3 = initializers.glorot_uniform()
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3,padding = 'same', activation='relu',kernel_initializer=ini, input_shape=(32,32,3)))
    model.add(MaxPooling2D(padding = 'same',strides=(1,1)))
    model.add(Conv2D(128, kernel_size=3,padding = 'same' ,activation='relu',kernel_initializer=ini1))
    model.add(MaxPooling2D(padding = 'same',strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(512,activation = 'relu',kernel_initializer=ini2))
    model.add(Dense(256,activation = 'relu',kernel_initializer=ini3))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    X_tr = Train_arr[0:50000,:]/255
    X_te = Train_arr[40000:50000,:]/255
    Y_tr = Y_ohe[0:50000,:]
    Y_te = Y_ohe[40000:50000,:]
    model.fit(X_tr, Y_tr, batch_size = 500,validation_data=(X_te, Y_te), epochs=22)
    y = model.predict(Test_arr/255)
    a = np.argmax(y,axis=1)
    for param in a:
        print(np.asscalar(param),file=open(output, "a"))
if __name__ == '__main__':
    train(*sys.argv[1:])