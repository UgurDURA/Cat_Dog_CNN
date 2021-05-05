import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import layers,Model,optimizer,regularizers
from keras.preprocessing import image
import random
from pathlib import Path



tf.__version__

train_datagen = ImageDataGenerator(
    rescale= 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set= train_datagen.flow_from_directory(
    r"Cat_Dog_data\Cat_Dog_data\train",
    
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

test_datagen= ImageDataGenerator(rescale=1./255)

test_set=test_datagen.flow_from_directory(
    r"Cat_Dog_data\Cat_Dog_data\test",
   
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

cnn= tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=16,kernel_size=3,activation='relu',input_shape=[150,150,3]))         #Convolution
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))                                                   #Max pooling


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[150,150,3]))         # 2nd Convolution layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))                                                   # 2nd Max pooling

cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[150,150,3]))         # 3th Convolution layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2)) 



cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(filter=256, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.3))

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])         #Compiling the CNN

cnn.fit(x=training_set,validation_data=test_set,epochs=15)              #Training the CNN

test_image= image.load_img()

















