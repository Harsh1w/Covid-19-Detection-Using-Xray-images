import numpy as np
import pandas as pd
import shutil
import os
print(os.listdir(os.getcwd()))
base_dir = os.getcwd()
work_dir=  "work/"
os.mkdir(work_dir)
base_dir_A ='covid19/'
base_dir_B ='covid19-/'

work_dir_A = "work/A/"
os.mkdir(work_dir_A)
work_dir_B = "work/B/"
os.mkdir(work_dir_B)
train_dir = os.path.join(work_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(work_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(work_dir, 'test')
os.mkdir(test_dir)

print("New directories for train, validation, and test created")
train_pos_dir = os.path.join(train_dir, 'pos')
os.mkdir(train_pos_dir)
train_neg_dir = os.path.join(train_dir, 'neg')
os.mkdir(train_neg_dir)

validation_pos_dir = os.path.join(validation_dir, 'pos')
os.mkdir(validation_pos_dir)
validation_neg_dir = os.path.join(validation_dir, 'neg')
os.mkdir(validation_neg_dir)

test_pos_dir = os.path.join(test_dir, 'pos')
os.mkdir(test_pos_dir)
test_neg_dir = os.path.join(test_dir, 'neg')
os.mkdir(test_neg_dir)

print("Train, Validation, and Test folders made for both A and B datasets")

i = 0
      
for filename in os.listdir(base_dir_A): 
       dst ="pos" + str(i) + ".jpg"
       src =base_dir_A + filename 
       dst =work_dir_A + dst 
          
       
# rename() function will 
       
# rename all the files 
       shutil.copy(src, dst) 
       i += 1 


       
j = 0
      
for filename in os.listdir(base_dir_B): 
       dst ="neg" + str(j) + ".jpg"
       src =base_dir_B + filename 
       dst =work_dir_B + dst 
          
       
# rename() function will 
       
# rename all the files 
       shutil.copy(src, dst) 
       j += 1       
        
print("Images for both categories have been copied to working directories, renamed to A & B + num")

fnames = ['pos{}.jpg'.format(i) for i in range(15)]
for fname in fnames:
    src = os.path.join(work_dir_A, fname)
    dst = os.path.join(train_pos_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['pos{}.jpg'.format(i) for i in range(15, 20)]
for fname in fnames:
    src = os.path.join(work_dir_A, fname)
    dst = os.path.join(validation_pos_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['pos{}.jpg'.format(i) for i in range(20, 25)]
for fname in fnames:
    src = os.path.join(work_dir_A, fname)
    dst = os.path.join(test_pos_dir, fname)
    shutil.copyfile(src, dst)
    


fnames = ['neg{}.jpg'.format(i) for i in range(15)]
for fname in fnames:
    src = os.path.join(work_dir_B, fname)
    dst = os.path.join(train_neg_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['neg{}.jpg'.format(i) for i in range(15, 20)]
for fname in fnames:
    src = os.path.join(work_dir_B, fname)
    dst = os.path.join(validation_neg_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['neg{}.jpg'.format(i) for i in range(20, 25)]
for fname in fnames:
    src = os.path.join(work_dir_B, fname)
    dst = os.path.join(test_neg_dir, fname)
    shutil.copyfile(src, dst)
    
print("Train, validation, and test datasets split and ready for use")
print('total training pos images:', len(os.listdir(train_pos_dir)))
print('total training neg images:', len(os.listdir(train_neg_dir)))
print('total validation pos images:', len(os.listdir(validation_pos_dir)))
print('total validation neg images:', len(os.listdir(validation_neg_dir)))
print('total test pos images:', len(os.listdir(test_pos_dir)))
print('total test meg images:', len(os.listdir(test_neg_dir)))

import keras
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        validation_dir,target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

print("Image preprocessing complete")

from keras import layers

import tensorflow as tf
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras import models
                                                          
img_shape=(150, 150, 3)
model = models.Sequential()
model.add(ResNet50(include_top=False, weights=None,
                       input_tensor=None, input_shape=img_shape) )
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax', name='fc1000'))



model.summary()
from keras import optimizers
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-5),
metrics=['acc'])
from PIL import Image
import sys
from PIL import Image
sys.modules['Image'] = Image


history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=200)
