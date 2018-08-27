import numpy as np
import cv2
import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout ,Flatten, Activation
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *


train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

# train_batches = ImageDataGenerator().flow_from_directory(
	# train_path, target_size=(224,224),
	# classes=['zakaz','info','nakaz','ostrzeg'],
	# batch_size = 32,
	# class_mode='categorical')

# valid_batches = ImageDataGenerator().flow_from_directory(
	# valid_path, target_size=(224,224),
	# classes=['zakaz','info','nakaz','ostrzeg'],
	# batch_size = 4,
	# class_mode='categorical')
	
# test_batches = ImageDataGenerator().flow_from_directory(
	# test_path, target_size=(224,224),
	# classes=['zakaz','info','nakaz','ostrzeg'],
	# batch_size = 3,
	# class_mode='categorical')
	#Preparation train data


# model = Sequential()
# model.add(Convolution2D(32, (3, 3), activation = 'relu',input_shape=(224,224,3), padding ="same"))
# model.add(MaxPooling2D(pool_size = (2,2)))

# model.add(Convolution2D(64,(2,2), activation = 'relu'), padding ="same")
# model.add(MaxPooling2D(pool_size = (2,2), data_format = "channels_first"))

# model.add(Flatten())
# model.add(Dense(256,activation = 'relu'))
# model.add(Dense(4,activation = 'softmax'))


# model.compile(loss = 'categorical_crossentropy',
				# optimizer = optimizers.RMSprop(lr=lr),
				# metrics = ['accuracy'])


# model.fit_generator(
    # train_generator,
    # steps_per_epoch=samples_per_epoch,
    # epochs=epochs,
    # validation_data=validation_generator,
    # callbacks=cbks,
    # validation_steps=validation_steps)					
				
				
#Parameters
img_width, img_height = 224, 224
epochs = 10
batch_size = 32
valid_batch_size = 10
samples_per_epoch = 100
validation_steps = 5
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 4
lr = 0.0004

#Przygotowanie danych
train_datagen = ImageDataGenerator(
    rescale=1. / 224,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1. / 224)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
	classes=['zakaz','info','nakaz','ostrzeg'],
    batch_size= batch_size,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    valid_path,
    target_size =(224, 224),
	classes=['zakaz','info','nakaz','ostrzeg'],
    batch_size = valid_batch_size,
    class_mode='categorical')

model = Sequential()
model.add(Convolution2D(nb_filters1, (conv1_size, conv1_size), input_shape=(img_width, img_height, 3), padding ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, (conv2_size, conv2_size), padding ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format = "channels_first"))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

#model.summary()

model.fit_generator(train_generator,
					steps_per_epoch =15,
					validation_data = valid_generator,
					epochs = epochs,	
					validation_steps = validation_steps )
					
#Result
model.save('model_saved1.h5')	

#Test
img = cv2.imread('data/test/ostrzeg/ostrzegawcze_012.jpg')
img = cv2.resize(img,(224,224))
img = np.reshape(img,[1,224,224,3])
prediction = model.predict(img)
print('Predicted: ',prediction)			