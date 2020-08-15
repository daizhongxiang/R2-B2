# Code from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

import GPy
import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf

import os

batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save the target ML model
model.save("mnist_saved_keras_model.h5")


# Below we find some random images to be used for attack/defense from those that are successfully predicted by the target ML model
# In particular, we randomly sample 100 (correctly predicted) images from each of the 10 classes
pred = model.predict(x_test, verbose=0)

all_img_inds = []
for c in range(10):
    groundtruth_class = c

    y_test_new = np.argmax(y_test, axis=1)
    pred_new = np.argmax(pred, axis=1)
    inds = np.nonzero((y_test_new == groundtruth_class) * (pred_new == groundtruth_class))[0]

    target_img_ind = np.random.choice(inds, 100)
    all_img_inds += list(target_img_ind)

pickle.dump(all_img_inds, open("img_inds_mnist.pkl", "wb"))






