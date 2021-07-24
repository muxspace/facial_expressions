
# coding: utf-8

# In[19]:
from __future__ import print_function
import time 
import sys
import requests
import cv2
import operator
import numpy as np
import scipy.misc
import os as os
from skimage import util, io
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Import library to display results
#import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
# Display images within Jupyter


# In[20]:

# Define  functions for applying transformations to the images

#Function to rotate images
def image_rotation(x, rotation_range, save_path):
    datagen = ImageDataGenerator(rotation_range=rotation_range)
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_path, save_prefix='cat', save_format='jpeg'):
             break  # otherwise the generator would loop indefinitely
   
#Function to shift height and width
def image_size_shift(x, w_shift, h_shift, save_path):
    datagen = ImageDataGenerator(width_shift_range=w_shift, height_shift_range=h_shift)
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_path, save_prefix='cat', save_format='jpeg'):
              break  # otherwise the generator would loop indefinitely

#Function to shift channel in the image
def image_channel_shift(x, c_shift, save_path):
    datagen = ImageDataGenerator(channel_shift_range=c_shift)
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_path, save_prefix='cat', save_format='jpeg'):
              break  # otherwise the generator would loop indefinitely

#Function to flip the image
def image_flip(x, h_flip, v_flip, save_path):
    datagen = ImageDataGenerator(horizontal_flip=h_flip, vertical_flip=v_flip)
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_path, save_prefix='cat', save_format='jpeg'):
              break  # otherwise the generator would loop indefinitely
    
#Function to shift features in the image
def image_featurewise(x, fwise_center, fwise_std_norm, save_path):
    datagen = ImageDataGenerator(featurewise_center=fwise_center, featurewise_std_normalization=fwise_std_norm)
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_path, save_prefix='cat', save_format='jpeg'):
                             break  # otherwise the generator would loop indefinitely

#Function to apply ZCA whitening to the image
def image_zca_whitening(x, zca_white, save_path):
    datagen = ImageDataGenerator(zca_whitening=zca_white)
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_path, save_prefix='cat', save_format='jpeg'):
                break  # otherwise the generator would loop indefinitely
    
#Function to apply samplewise transformations
def image_samplewise(x, swise_center, swise_std_norm, save_path):
    datagen = ImageDataGenerator(samplewise_center=swise_center, samplewise_std_normalization=swise_std_norm)
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_path, save_prefix='cat', save_format='jpeg'):
                break  # otherwise the generator would loop indefinitely

#Function to rescale the image
def image_rescale(x, resc, shear_range, zoom_range, h_flip, save_path):
    datagen = ImageDataGenerator(
        rescale=resc,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=h_flip)
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_path, save_prefix='cat', save_format='jpeg'):
                break  # otherwise the generator would loop indefinitely

#Flip the color of pixels (black to white and vice versa)   
def image_invert(image):
    return np.invert(image)

#Create blurred images
def image_blurred(image, sig):
    return ndimage.gaussian_filter(image, sigma=sig)
    


# In[22]:

#Apply the transformations on all the images in a particular folder. New images will be created in aug_images sub-folder
def process_images(main_folder):
    
    #Some of the concepts were adapted from http://machinelearningmastery.com/image-augmentation-deep-learning-keras/
    #Treating individual transformation arguments as options for transformations
    #Prepare list of images to be processed for transformations

    # Folder where original images exist
    file_list = os.listdir(main_folder)
    # print(file_list)

    # Change current working directory and create a sub-folder called aug_images to save augmented images
    working_location = os.chdir(main_folder)
    working_location = os.getcwd()
		
    dirname = 'aug_images'
    if not os.path.exists(dirname):
       os.makedirs(dirname, mode=0777)

    # Directory where augmented (new) images will be saved
    save_path = main_folder + dirname

    for filename in file_list:
         basename = os.path.splitext(filename)[0] # filename without extension
         img = load_img(filename)  # this is a PIL image
         image = io.imread(filename) # this is a Numpy array
         x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
         x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

         #Run 10 transformation on each image and create 10 new images

         rotation_range=40
         image_rotation(x, rotation_range, save_path)
    
         w_shift=0.2; h_shift=0.2
         image_size_shift(x, w_shift, h_shift, save_path)
    
         c_shift=0.2
         image_channel_shift(x, c_shift, save_path)
    
         h_flip=True; v_flip=True
         image_flip(x, h_flip, v_flip, save_path)
    
         fwise_center=False; fwise_std_norm=False  
         image_featurewise(x, fwise_center, fwise_std_norm, save_path)
    
         zca_white=False
         image_zca_whitening(x, zca_white, save_path)
    
         swise_center=False; swise_std_norm=False
         image_samplewise(x, swise_center, swise_std_norm, save_path)
    
         resc = 1./255; shear_range=0.2; zoom_range=0.2; h_flip=True
         image_rescale(x, resc, shear_range, zoom_range, h_flip, save_path)
    
         im_out = image_invert(image)
         io.imsave(save_path+"\\"+ basename + "_inverted.jpeg", im_out)

         sig=5 #sigma
         im_out = image_blurred(image, sig)
         io.imsave(save_path+"\\"+ basename + "_blurred.jpeg", im_out)


# In[ ]:

# The code can be run by changing the folder at line:
# main_folder = "C:\Images_Week_10"
# process_images(main_folder = "C:\Images_Week_10\")
if __name__ == "__main__":
    process_images(sys.argv[1])

# In[ ]:



