# import the libraries as shown below
# we are using VGG16 and VGG19
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
print(tf.__version__) #here check the tf version

# resize all train images
IMAGE_SIZE = [224, 224]

# we are train our model with 40 images
# we are providing here three folder for training 

train_path = 'CancerDatasets/train'
valid_path = 'CancerDatasets/test'

# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we are providing the "IMAGE_SIZE" and also the RGB dimenssion 3
# Here we will be using imagenet weights for categorises the images
# Basically VGG16 used for classify thousand different categorises  of images that's why "include_top = False". 
# Because we are training 2 categorises images. So we are droping the last column
# Also droping the first column because we are providing the "IMAGE_SIZE"

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
# Putting the for loop and make sure that all the layer shouldn't be train

for layer in vgg.layers:
    layer.trainable = False

# how many classes or categorises in my train dataset
folders = glob('CancerDatasets/train/*')

# After the droping the first & last layer
# We just the the Flatten layer
# Flatten layer basically converts the features map to a single column

x = Flatten()(vgg.output)

# Finally adding the last layer(Dense) and also provide the folder length. We need how many categories we have.
# Then we are using the activation function called "softmax"

prediction = Dense(len(folders), activation='softmax')(x)


# Then combine the vgg output and input this will create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view  the model strcture 
# Absorve the last layer(Dense) it is just having 2 output categorises
# Because our dataset  have 2 categorises 

model.summary()

# tell the model what cost and optimization method to use
# Adam optimizer provide an optimization algorithm which can handle sparse gradients on noisy problems.

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
# we upload dataset using ImageDataGenerator which will help in train and test dataset.

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# we insert images using flow_from_directory . For batch size, at a time 32 images will demand for training


training_set = train_datagen.flow_from_directory('CancerDatasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('CancerDatasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# fit the model
# we use fit_generator from library for running it. we will use train data set for running and test data for validation.

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=8,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# save it as a h5 file

import tensorflow as tf

from keras.models import load_model

model.save('model_vgg19.h5')

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
model=load_model('model_vgg19.h5')
img=image.load_img('CancerDatasets/val/normal/normal (1).jpg', target_size=(224, 224))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)
malignant=classes[0,0]
normal=classes[0,1]

if(malignant==1):
    print('P: malignant')
else:
    print('P: normal')
