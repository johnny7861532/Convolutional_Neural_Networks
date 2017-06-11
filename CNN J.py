#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 16:11:33 2017

@author: johnnyhsieh
"""

#building the CNN 
#import the library we need to used on CNN
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Dropout

#init the CNN
classifier = Sequential()
#step1 convolution
leakyrelu = LeakyReLU(alpha=0.2)
#32=filter 3,3 = 3x3 input_shape 3= rgb color if mono = 1 64x64 = size of image
classifier.add(Convolution2D(32,3,3,border_mode = 'valid', input_shape = (128,128,3),activation = leakyrelu))
#step2 Maxpooling
classifier.add(MaxPooling2D(pool_size = (2,2))) 

#second time convultion
#because image has already load so we don't need to info the input_shape
classifier.add(Convolution2D(64,3,3,border_mode = 'valid',activation = leakyrelu))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(128,3,3,border_mode = 'valid',activation = leakyrelu))
classifier.add(MaxPooling2D(pool_size = (2,2)))


#step3 flatten
classifier.add(Flatten())
#step4 full connection
classifier.add(Dense(200,activation=leakyrelu,kernel_initializer='uniform'))
classifier.add(Dropout(p = 0.5))
classifier.add(Dense(100,activation=leakyrelu,kernel_initializer='uniform'))
classifier.add(Dropout(p = 0.5))
classifier.add(Dense(1,activation = 'sigmoid'))

#compiling CNN
classifier.compile('adam','binary_crossentropy',metrics = ['accuracy'])
#input image data set
#add some random image data set to prevent the over fitting
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#dataset/train_set is the path of your dataset setps_per_epoch = train number of your data
#validation_steps = test number of your data
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=3,
        validation_data=test_set,
        validation_steps=2000
        )


import numpy as np
from keras.preprocessing import image
#loading image and it's dimetion must fit to the model we have train ex:128
test_image = image.load_img('dataset/single_prediction/cat.4053.jpg',target_size = (128,128))
#turn test_image into 3D array that can fit out model
test_image = image.img_to_array(test_image)
#but if u run the model and it's will required 4D, so u need to using numpy to add one dimesion
test_image = np.expand_dims(test_image,axis = 0)
request = classifier.predict(test_image)
#need to know what 0 and 1 represent we need to find out from train_set
result_represent = training_set.class_indices

if request[0][0] == 1:
    predict = 'Dog'
else:
    predict = 'Cat'

predict

from keras.models import load_model 
classifier.save_weights('CatandDogCNN.h5')
