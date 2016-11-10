
# coding: utf-8

# Author:  Alex Satz
# code: 906
# Date: Nov 6 2016

# In[12]:


import numpy as np
from skimage.io import imread, imsave


# In[19]:

from IPython.display import Image


# ## Please use the function below
# 
# augphotos = RunAllAugmentations(img, d)
# 
# The above will convert an image to 14 different images, and save each in the list 'augphotos'

# In[11]:

from skimage.util import random_noise
import re
from skimage import transform


# In[78]:

def augmentVar(img, v = .01):
    return random_noise(img, mode='gaussian', var = v)


# In[76]:

def FlipH(img):
    return np.fliplr(img)


# In[37]:

def augmentZoom(img, p1x, p1y, p2x, p2y):
        h = len(img)
        w = len(img[0])

        crop_p1x = max(p1x, 0)
        crop_p1y = max(p1y, 0)
        crop_p2x = min(p2x, w)
        crop_p2y = min(p2y, h)

        cropped_img = img[crop_p1y:crop_p2y, crop_p1x:crop_p2x]

        x_pad_before = -min(0, p1x)
        x_pad_after  =  max(0, p2x-w)
        y_pad_before = -min(0, p1y)
        y_pad_after  =  max(0, p2y-h)

        padding = [(y_pad_before, y_pad_after), (x_pad_before, x_pad_after)]

        padded_img = np.pad(cropped_img, padding, 'constant')
        return transform.resize(padded_img, (h,w))


# In[107]:

def zoom1(img):
    img = augmentZoom(img, 40, 40, -10, -10)
    return img

def zoom2(img):
    img = augmentZoom(img, 20, 20, -10, -10)
    return img

def noise1(img):
    augmentVar(img, v = .01)
    return img

def noise2(img):
    augmentVar(img, v = .005)
    return img

def FlipandZoom1(img):
    img = FlipH(img)
    img = augmentZoom(img, 40, 40, -10, -10)
    return img

def FlipandZoom2(img):
    img = FlipH(img)
    img = augmentZoom(img, 20, 20, -10, -10)
    return img

def FlipandNoise1(img):
    img = FlipH(img)
    img = augmentVar(img, v = .01)
    return img

def FlipandNoise2(img):
    img = FlipH(img)
    img = augmentVar(img, v = .005)
    return img


def ZoomandNoise1(img):
    img = augmentZoom(img, 40, 40, -10, -10)
    img = augmentVar(img, v = .005)
    return img

def ZoomandNoise2(img):
    img = augmentZoom(img, 20, 20, -10, -10)
    img = augmentVar(img, v = .005)
    return img

def FlipandZoomandNoise1(img):
    img = FlipH(img)
    img = augmentZoom(img, 40, 40, -10, -10)
    img = augmentVar(img, v = .005)
    return img

def FlipandZoomandNoise2(img):
    img = FlipH(img)
    img = augmentZoom(img, 20, 20, -10, -10)
    img = augmentVar(img, v = .005)
    return img

def FlipandZoomandNoise3(img):
    img = FlipH(img)
    img = augmentZoom(img, 40, 40, -10, -10)
    img = augmentVar(img, v = .01)
    return img

def FlipandZoomandNoise4(img):
    img = FlipH(img)
    img = augmentZoom(img, 20, 20, -10, -10)
    img = augmentVar(img, v = .01)
    return img

d = {'1':zoom1, '2':zoom2, '3':noise1, '4':noise2, '5':FlipandZoom1, '6':FlipandZoom2, '7':FlipandNoise1, '8':FlipandNoise2
    ,'9':ZoomandNoise1, '10':ZoomandNoise2, '11':FlipandZoomandNoise1, '12':FlipandZoomandNoise2, '13':FlipandZoomandNoise4
    ,'14':FlipandZoomandNoise4}

def RunAllAugmentations(img, d):
    l1 = []
    for key, value in d.iteritems():    
        l1.append(value(img))
    return l1
    

