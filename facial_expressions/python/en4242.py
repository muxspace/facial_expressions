'''
For the code to import pandas correctly using the Openface Docker image
the code needs to run from the /root/openface/... directory.
'''
import cv2
import os
import numpy as np
np.set_printoptions(precision=2)
import openface
import pandas as pd
import csv


# Set the image directories. These can be overridden in the function calls.
img_directory = '/root/openface/project/data/expression/'
out_directory = '/root/openface/project/data/augmented/'
legend_out = '/root/openface/project/data/augmented/'
pa_legend = '/root/openface/project/data/expression/legend.csv'


# Creating the output file legend.csv
with open(out_directory + 'legend.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['usr.id', 'image', 'emotion'])


# Function for saving the file and writing the legend file.
def save_image(img, name, emotion, out_folder = out_directory, legend = legend_out):
    with open(legend + 'legend.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([usr, name, emotion])
    cv2.imwrite(out_folder + name, img)


# Performs the horizontal flip of the image.
def hflip_img(image, emotion, in_folder = img_directory):
    # Load raw image file into memor
    img = cv2.imread(in_folder + image)
    res = cv2.flip(img, 1) # Flip the image
    save_image(res, 'hflip'+image, emotion)


# Performs a vertical flip of the image.
def vflip_img(image, emotion, in_folder = img_directory):
    # Load raw image file into memor
    img = cv2.imread(in_folder + image)
    res = cv2.flip(img, 0) # Flip the image
    save_image(res, 'vflip'+image, emotion)


# Rotates the image given a specific number of degrees, positive is clockwise
# negative is counterclockwise.
def rotate_img(image, emotion, angle, in_folder = img_directory):
    img = cv2.imread(in_folder + image, 0)
    rows,cols = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    save_image(dst, str(angle) + 'rotate' + image, emotion)


# Translates the image horizontally and vertically, postivie is down and right
# negative is up and left.
def shift_img(image, emotion, x, y, in_folder = img_directory):
    img = cv2.imread(in_folder + image, 0)
    rows,cols = img.shape

    M = np.float32([[1,0,x],[0,1,y]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    save_image(dst, str(x)+ '_' + str(y) + 'shift' + image, emotion)


# Blurs the image using the average value from the 5X5 pixle square surrounding
# each pixel.
def blur_img(image, emotion, size = 5, in_folder = img_directory):
    img = cv2.imread(in_folder + image, 0)
    blur = cv2.blur(img,(size,size))
    save_image(blur, 'blur' + image, emotion)


# Blurs the image using Gaussian weights from the 5X5 pixle square surrounding
# each pixel.
def gauss_img(image, emotion, size = 5, in_folder = img_directory):
    img = cv2.imread(in_folder + image, 0)
    blur = cv2.GaussianBlur(img,(size,size), 0)
    save_image(blur, 'gauss' + image, emotion)


# Applys a bilateral filter that sharpens the edges while bluring the other areas.
def bilateral_img(image, emotion, size = 5, in_folder = img_directory):
    img = cv2.imread(in_folder + image, 0)
    blur = cv2.bilateralFilter(img,9,75,75)
    save_image(blur, 'bilat' + image, emotion)



if __name__ == '__main__':
    # Loading the basic legend file befor augmentation
    legend = pd.read_csv(pa_legend)
    files = legend['image']
    emotion = legend['emotion']
    usr = 'en4242'
    i = 0
    for f in files: # Running the augmentations
        emo = emotion[i]
        hflip_img(f, emo)
        vflip_img(f, emo)
        rotate_img(f, emo, 15)
        rotate_img(f, emo,-15)
        rotate_img(f, emo, 30)
        rotate_img(f, emo, -30)
        shift_img(f, emo, 50, 50)
        shift_img(f, emo, -50, -50)
        blur_img(f, emo)
        gauss_img(f, emo)
        bilateral_img(f, emo)
        i += 1
