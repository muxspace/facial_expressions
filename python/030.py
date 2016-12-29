# -*- coding: utf-8 -*-
"""
Created on Sun Nov 06 20:57:23 2016

@author: Parshu Rath
"""

import os
import sys
#from IPython.display import Image
import matplotlib.pyplot as plt
#import time 
import numpy as np
import scipy
import scipy.misc
from scipy import ndimage
from scipy import misc
from skimage import transform as tf
import math
import re

""" 
Define transformation functons.
"""

#Flip the image 
def flipt(f):
    return(np.flipud(f))
    
#Rotate at some random angle transform 
def rotatet(f):
    angle = math.floor(np.random.uniform(-90, 90, 1))
    return (ndimage.rotate(f, angle, reshape=False))

#Add some random noise transform 
def noisyt(f):
    noisy = f + 0.9 * f.std() * np.random.random(f.shape)
    return (f)

#Add Gaussian transform Filter
def gausst(f):
    f1 = ndimage.gaussian_filter(f, sigma=3)
    return(f1)

#Add Uniform transform Filter
def uniformt(f):
    f1 = ndimage.uniform_filter(f, size=10)
    return(f1)

#Add Median transform Filter
def mediant(f):
    f1 = ndimage.median_filter(f, size=10)
    return(f1)

#Add Fourrier ellipsoid transform Filter
def fouriert(f):
    f1 = ndimage.fourier_ellipsoid(f, size=1.0)
    return(f1)

#Add Fourrier uniform transform Filter
def fourierut(f):
    f1 = ndimage.fourier_uniform(f, size=1.25)
    return(f1)

#Add Affine transform Filter
def affinet(f):
    H = np.array([[1.4,0.05,-100],[0.05,1.4,-100],[0,0,1]])
    f1 = ndimage.affine_transform(f,H[:2,:2],(H[0,2],H[1,2]))
    return(f1)

#Add Geometric transform Filter
def shift_func(output_coordinates):
     return (output_coordinates[0] - 10.5, output_coordinates[1] - 10.5)
def geomt(f):
    H = np.array([[1.4,0.05,-100],[0.05,1.4,-100],[0,0,1]])
    f1 = ndimage.geometric_transform(f, shift_func)
    return(f1)
    
"""
Below is an example of applying the above transformations to images in a directory 
and saving the transformed images.
"""    
#List of trsansformations
transforms = [flipt, rotatet, noisyt, gausst, uniformt, mediant, fouriert, fourierut, affinet, geomt]
#Source image directory
imgdir = r'C:\pr\CUNY\MSDA Fall 2016\DATA 622\Kaggle Project\img10'
#Get names of the images
file_list = os.listdir(imgdir)
#Directory to save the transformed images
img_savedir = r'C:\pr\CUNY\MSDA Fall 2016\DATA 622\Kaggle Project\trans_img'

#Transform the files and save.

#import re
for f in file_list:
    #use one transformation at a time
    for i in range(0, len(transforms)):
        #read the image file
        f1 = misc.imread(''.join([imgdir, '\\', f]))
        #transform the image
        f1t = transforms[i](f1)
        #save the image after adding '-[i]t' in the file name
        scipy.misc.imsave(''.join([img_savedir, '\\', re.sub(r'.jpg', '-', f), str(i),'t.jpg']), f1t)

