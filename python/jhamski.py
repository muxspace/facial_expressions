#!/usr/bin/env python

import csv
import os
import math

from skimage import io
from skimage import transform as tf

# Import legend.csv and save my (jhamski) image files to a list
script_dir = os.path.dirname(__file__)
full_path_csv = os.path.join(script_dir, '../data/legend.csv')

ifile  = open(full_path_csv, "rb")
reader = csv.reader(ifile)

#rownum = 0
jhamski_pics = []
for row in reader:
    if row[0] == 'jhamski':
        pic_name = row[1]
        jhamski_pics.append(pic_name)

ifile.close()

#function to read in images

def read_image(image_name, script_dir):

    image_path = os.path.join(script_dir, '../images/%s' % image_name)
    img = io.imread(image_path)
    return img

def save_image(img, name, trans_type):
    name = os.path.splitext(name)[0]
    io.imsave(name + "_" + trans_type + '.png', img,)


#setup image transformation functions

def warp1(img, name):
    tform = tf.SimilarityTransform(scale=1, rotation=math.pi / 4,
                               translation=(img.shape[0] / 2, -100))
    af_img = tf.warp(img, tform)

    save_image(af_img, name, 'warp1')

def warp2(img, name):
    tform = tf.SimilarityTransform(scale=1, rotation=math.pi / -4,
                               translation=(img.shape[0] / 5, 100))
    af_img = tf.warp(img, tform)

    save_image(af_img, name, 'warp2')


# read in each image in the in the list, transform, and save
# swap out testing length 5 for len(jhamski_pics)
for i in range(0, 5):
    img = read_image(jhamski_pics[i], script_dir)

    warp1(img, jhamski_pics[i])
    warp2(img, jhamski_pics[i])
