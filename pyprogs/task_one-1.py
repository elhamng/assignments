#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 20:25:08 2021

@author: elham
"""
## all lib 
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import tensorflow as tf
from keras.preprocessing.image import img_to_array,array_to_img
import os
import glob
import random

#%matplotlib inline
np.random.seed(4)
tf.random.set_seed(4)


## all functionas 

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def resize_image(image,shape=True):
    width, height = image.size
    shape = True
    if width == 480 and height == 270:
        return image, shape
    else:
        new_image = image.resize((480, 270))
        shape = False
        return new_image, shape


def crop_image (image, new_width, new_height):
    new_width = new_width
    new_height = new_height
    width,height = image.size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = image.crop((left, top, right, bottom))
    return im 

def sample_image(image,frame_dim=(80,80),seed=42,n=3,padding='VALID'):
    if n > (image.shape[0] * image.shape[1]) // (frame_dim[0] * frame_dim[1]):
        padding = 'SAME' 

    patches = tf.image.extract_patches(tf.reshape(image,shape=(-1,*image.shape)),
                         [1,*frame_dim,1],
                         [1,*frame_dim,1],
                         [1,1,1,1],padding=padding)

    patches_res = tf.reshape(patches,shape=(-1,*frame_dim,image.shape[2]))

    ixs = tf.reshape(tf.range(patches_res.shape[0],dtype=tf.int64),shape=(1,-1))
    
    ixs_sampled = tf.random.uniform_candidate_sampler(ixs,
                                                  patches_res.shape[0],n,
                                           unique=True,range_max=patches_res.shape[0],seed = 4)

    ixs_sampled_res = tf.reshape(ixs_sampled.sampled_candidates,shape=(n,1))
    return patches_res , ixs_sampled_res


image_list = []
sample_image_list = []

for filename in glob.glob('sample_frames/*.jpeg'): #assuming jpeg
    image=Image.open(filename)
    image_to_list = resize_image(image,shape=True)
    crop_to_list = crop_image(image_to_list[0], new_width = 270, new_height = 270)
    image_list.append(crop_to_list)
    croped_image = load_image_into_numpy_array(crop_to_list)
    patches,indxes = sample_image(croped_image)
    sample_one = indxes[0,0].numpy()
    sample_two = indxes[1,0].numpy()
    sample_three = indxes[2,0].numpy()
    patche_one = patches[sample_one]
    patche_two = patches[sample_two]
    patche_three = patches[sample_three]
    sample_image_list.extend([patche_one, patche_two, patche_three])
    


image = sample_image_list[4]
print(image.shape)
plt.imshow(image)


random.Random(4).shuffle(sample_image_list)

train_data = sample_image_list[:312]  ## 80% train 
test_data = sample_image_list[312:]    ## 20% test 

print(test_data[3].shape)
plt.imshow(test_data[3])




##### save train and test images  in files     




pathtest = 'testdata'
for i in range(0, len(test_data)):
    
    
    img = array_to_img(test_data[i])
    img.save(os.path.join(pathtest , 'test_number{0}.jpeg'.format(i)))
    


pathtrain = 'traindata'

for i in range(0, len(train_data)):
    
    
    img = array_to_img(train_data[i])
    img.save(os.path.join(pathtrain , 'train_number{0}.jpeg'.format(i)))
    
plt.imshow(array_to_img(test_data[1]))




