# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 12:23:38 2016

@author: Xingjia Wu

To run:
indir = r'.\test' # Directory having images for augmentation
newdir = r'.\aug' # Directory for saving augmented images. Will create the folder if not exist.

picaugment(indir, newdir)

"""

import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
def picaugment(indir, newdir, augsize = 10):
    
    # Using keras ImageDataGenerator to generate random images
    datagen = ImageDataGenerator(
        featurewise_std_normalization=False,
        rotation_range = 20,
        width_shift_range = 0.10,
        height_shift_range = 0.10,
        shear_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True)
    
    piclist = os.listdir(indir)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    for i in range(0,len(piclist)):
        img = load_img(os.path.join(indir, piclist[i]))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        j = 0
        for batch in datagen.flow(x, batch_size = 1, save_to_dir = newdir, 
                              save_prefix = piclist[i].split('.')[0]):
            j += 1
            if j >= augsize:
                break
