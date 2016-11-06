from scipy import ndimage
from scipy import misc
import numpy as np
from glob import glob
from collections import Iterable

# create a list of all images in directory
images = glob('*.jpg')
images.sort()

def apply_transformations(images):
    for i in range(len(images)):
        img = misc.imread(images[i])
        # flip up-down
        flip_ud_img = np.flipud(img)
        misc.imsave('flipped_%06d.jpg' %i, flip_ud_img)
        # rotate 30 degrees
        rotate30_img = ndimage.rotate(img, 30)
        misc.imsave('rotated30_%06d.jpg' %i, rotate30_img)
        # rotate 45 degrees
        rotate45_img = ndimage.rotate(img, 45)
        misc.imsave('rotated45_%06d.jpg' %i, rotate45_img)
        # rotate 60 degrees
        rotate60_img = ndimage.rotate(img, 60)
        misc.imsave('rotated60_%06d.jpg' %i, rotate60_img)
        # rotate 90 degrees
        rotate90_img = ndimage.rotate(img, 90)
        misc.imsave('rotated90_%06d.jpg' %i, rotate90_img)
        # rotate  105 degrees
        rotate105_img = ndimage.rotate(img, 105)
        misc.imsave('rotated105_%06d.jpg' %i, rotate105_img)
        # rotate 120 degrees
        rotate120_img = ndimage.rotate(img, 120)
        misc.imsave('rotated120_%06d.jpg' %i, rotate120_img)
        # rotate 135 degrees
        rotate145_img = ndimage.rotate(img, 135)
        misc.imsave('rotated135_%06d.jpg' %i, rotate145_img)
        # rotate 150 degrees
        rotate160_img = ndimage.rotate(img, 150)
        misc.imsave('rotated150_%06d.jpg' %i, rotate160_img)
        # blurring
        blurred_img = ndimage.gaussian_filter(img, 4)
        misc.imsave('blurred_%06d.jpg' %i, blurred_img)