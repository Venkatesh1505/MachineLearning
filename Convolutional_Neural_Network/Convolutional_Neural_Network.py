# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:32:06 2019

@author: Venkatesh Ravichandran

Convolutional Neural Networks

"""

#importing required libraries and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Building the CNN
classifier = Sequential()

#Adding the convolutional layer
classifier.add(Convolution2D(34, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Adding the Pooling layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding the second convolutional layer
classifier.add(Convolution2D(34, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flattening Layer
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

from  scipy import ndimage

classifier.fit_generator(training_set,
                         steps_per_epoch=250,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=62)

