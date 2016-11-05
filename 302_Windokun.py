# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 20:46:49 2016
@author: Jare_2

Code adapted from https://github.com/aleju/ImageAugmenter
Requires the following per the developer:
    There is no pip-installer or setup.py for this class. 
    Simply copy ImageAugmenter.py to your project. Then import it, 
    create a new ImageAugmenter object and use ImageAugmenter.augment_batch()
    to augment an array of images. The function expects a numpy array
    (of dtype numpy.uint8 with values between 0 and 255) of images. 
    The expected shape of that array is any of the following:
"""

from ImageAugmenter import ImageAugmenter
from scipy import misc
import numpy as np
import os




def get_imlist(path):
    # Returns a list of filenames for all jpg images in a directory. 
   
    return ([os.path.join(path, f) for f in os.listdir(path)])



def augmentImage(imageFile):
    
    image = misc.imread(imageFile)
    height = image.shape[0]
    width = image.shape[1]
    augmenter = ImageAugmenter(width, height, # width and height of the image (must be the same for all images in the batch)
                           hflip=True,    # flip horizontally with 50% probability
                           vflip=False,    # flip vertically with 50% probability
                           scale_to_percent=1.1, # scale the image to 70%-130% of its original size
                           scale_axis_equally=True, # allow the axis to be scaled unequally (e.g. x more than y)
                           rotation_deg=10,    # rotate between -25 and +25 degrees
                           shear_deg=10,       # shear between -10 and +10 degrees
                           translation_x_px=5, # translate between -5 and +5 px on the x-axis
                           translation_y_px=5  # translate between -5 and +5 px on the y-axis
                           )
                           
    augmented_images = augmenter.augment_batch(np.array([image], dtype=np.uint8))
    #plt.imshow(augmented_images.squeeze(), cmap="gray")
    return (augmented_images)
                           



def imgAugmenter (image_directory, save_directory, num_augments):
      
    filelist = get_imlist(image_directory)
    filelist = filelist
    
    for f in  filelist[0:10]: 
            (filepath,filename) = os.path.split(f)
            for i in range(0, num_augments):
               aug_file = augmentImage(f) 
               savefilename = os.path.join(save_directory, 'aug_' + str(i) + '_' + filename)
               misc.toimage(aug_file.squeeze(), cmin=0.0, cmax=1.0).save(savefilename)
               print ("file save as......", savefilename)
            
# Example of use
    
image_directory = r'C:\Users\Jare_2\OneDrive\WorkDocs\CUNY\622\Week 10\code\Resize'
save_directory = r'C:\Users\Jare_2\OneDrive\WorkDocs\CUNY\622\Week 10\code\Augmented_images'

imgAugmenter (image_directory, save_directory, 10)                 





