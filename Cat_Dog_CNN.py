import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

train_datagen = ImageDataGenerator(
    rescale= 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set= train_datagen.flow_from_directory(
    'Cat_Dog_data\Cat_Dog_data\train',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

test_datagen= ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory(
    'Cat_Dog_data\Cat_Dog_data\test',
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

cnn= tf.keras.models.sequential()

cnn.add(tf.keras.layers.Conv2D(filters=16,kernel_size=3,activation='relu',input_shape=[150,150,3]))         #Convolution
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))                                                   #Max pooling


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[150,150,3]))         # 2nd Convolution layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))                                                   # 2nd Max pooling

cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[150,150,3]))         # 3th Convolution layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2)) 

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


















