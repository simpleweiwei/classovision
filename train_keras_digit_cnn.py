"""
Trains a CNN to classify digits
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
from math import ceil

import utils

model_path = r'saved_networks/cnn_alldignan_DONOTUSE.h5'

input_x_data = r'data/x_digit_data_unbalanced.npy'
input_y_data = r'data/y_digit_data_unbalanced.npy'

batch_size = 128
num_classes = 11
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
channels_first = False
input_shape = (img_rows, img_cols, 1)

# the data, split between train and test sets
x_data = utils.load_nd_array(input_x_data, (-1, img_rows, img_cols, 1))
y_data = utils.load_2d_array(input_y_data)

samples = np.shape(y_data)[0]

shuffle_ind = np.random.permutation(samples)

x_data = np.take(x_data,shuffle_ind,axis=0,out=x_data)
y_data = np.take(y_data,shuffle_ind,axis=0,out=y_data)

train_split = ceil(samples*0.7)
x_train = x_data[0:train_split,:]
y_train = y_data[0:train_split,:]
x_test = x_data[train_split:samples,:]
y_test = y_data[train_split:samples,:]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

try:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
except Exception as ex:
    print('Training failed, trying to save...')
    model.save(model_path)
    print('Save success')

try:
    model.save(model_path)
    print('Model trained and saved successfully!')
except Exception as ex:
    print('Model save failed: {0}'.format(ex))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Training finish')