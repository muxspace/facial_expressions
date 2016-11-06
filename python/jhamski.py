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
    script_dir = os.path.dirname(__file__)

    full_path_images = os.path.join(script_dir, '../images/', name + "_" + trans_type + '.png')
    io.imsave(full_path_images, img)
    #io.imsave(name + "_" + trans_type + '.png', img,)


#setup image transformation functions

def warp12(img, name):
    tform = tf.SimilarityTransform(scale=1, rotation=math.pi / 4,
                               translation=(img.shape[0] / 2, -100))
    af_img = tf.warp(img, tform)
    save_image(af_img, name, 'warp1')

    af_img2 = tf.warp(img, tform.inverse)
    save_image(af_img2, name, 'warp2')

def warp34(img, name):
    tform = tf.AffineTransform(shear=math.pi / -3.6)

    af_img3 = tf.warp(img, tform)
    save_image(af_img3, name, 'warp3')

    af_img4 = tf.warp(img, tform.inverse)
    save_image(af_img4, name, 'warp4')

def rotate180(img, name):
    img = tf.rotate(img, angle = 180)
    save_image(img, name, 'rotate180')

def rotate90(img, name):
    img = tf.rotate(img, angle = 90)
    save_image(img, name, 'rotate90')

def rotate270(img, name):
    img = tf.rotate(img, angle = 270)
    save_image(img, name, 'rotate270')

def rotate30(img, name):
    img = tf.rotate(img, angle = 30)
    save_image(img, name, 'rotate30')

def rotate120(img, name):
    img = tf.rotate(img, angle = 120)
    save_image(img, name, 'rotate120')

def rotate300(img, name):
    img = tf.rotate(img, angle = 300)
    save_image(img, name, 'rotate300')

def rotate210(img, name):
    img = tf.rotate(img, angle = 210)
    save_image(img, name, 'rotate210')


# read in each image in the in the list, transform, and save
# swap out testing length 5 for len(jhamski_pics)
for i in range(0, len(jhamski_pics)):
    img = read_image(jhamski_pics[i], script_dir)

    warp12(img, jhamski_pics[i])
    warp34(img, jhamski_pics[i])
    rotate180(img, jhamski_pics[i])
    rotate90(img, jhamski_pics[i])
    rotate270(img, jhamski_pics[i])
    rotate30(img, jhamski_pics[i])
    rotate120(img, jhamski_pics[i])
    rotate300(img, jhamski_pics[i])
    rotate210(img, jhamski_pics[i])
