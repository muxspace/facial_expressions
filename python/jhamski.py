#!/usr/bin/env python

import csv
import os

from skimage import io


#import pandas as pd

# Import legend.csv and save my (jhamski) image files to a list
script_dir = os.path.dirname(__file__)
full_path_csv = os.path.join(script_dir, '../data/legend.csv')

ifile  = open(full_path_csv, "rb")
reader = csv.reader(ifile)

rownum = 0
jhamski_pics = []

for row in reader:
    if row[0] == 'jhamski':
        pic_name = row[1]
        jhamski_pics.append(pic_name)

ifile.close()

#read in images

def read_image(image_name, script_dir):

    image_path = os.path.join(script_dir, '../data/%s' % image_name)
    img = io.imread(image_path)
    return img


#setup image transformation functions

# read in each image in the in the list, transform, and save

for i in range(0, len(jhamski_pics)):
    read_image(jhamski_pics[i])
