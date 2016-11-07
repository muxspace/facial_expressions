#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join

#Image rotation.
def rotate_transform(image, degrees):
    img = cv2.imread(image, 0)
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return (dst)

#Affine transformation.
def affine_transform(image, points1, points2):
    img = cv2.imread(image)
    rows, cols, ch = img.shape
    pts1 = np.float32(points1) #upper-left, upper-right, bottom-left
    pts2 = np.float32([points2]) # New locations
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return(dst)

#Perspective transformation.
def perspective_transform(image, points1, points2):
    img = cv2.imread(image)
    rows, cols, ch = img.shape
    pts1 = np.float32(points1)
    pts2 = np.float32(points2)
    M = cv2.getPerspectiveTransform(points1, points2)
    dst = cv2.warpPerspective(img, M, (350, 350))
    return (dst)

# The main function to transform images. Pass a list of image files to this function.
# This function should be run from the the folder containing the images.
def transformImages(imageList):
    for file in imageList:
        filename, file_extension = os.path.splitext(file)
        print('Processing image : ' + file)
        if file.startswith('.') or file.endswith('.py') or file.endswith('.DS_Store'):
            continue

        dst = rotate_transform(file, 45)
        cv2.imwrite(filename + '_rot_45' + file_extension, dst)

        dst = rotate_transform(file, 90)
        cv2.imwrite(filename + '_rot_90' + file_extension, dst)

        dst = rotate_transform(file, 135)
        cv2.imwrite(filename + '_rot_135' + file_extension, dst)

        dst = rotate_transform(file, 180)
        cv2.imwrite(filename + '_rot_180' + file_extension, dst)

        dst = rotate_transform(file, 225)
        cv2.imwrite(filename + '_rot_225' + file_extension, dst)

        dst = rotate_transform(file, 270)
        cv2.imwrite(filename + '_rot_270' + file_extension, dst)

        dst = rotate_transform(file, 315)
        cv2.imwrite(filename + '_rot_315' + file_extension, dst)

        points1 = [[10, 10], [80, 20], [5, 80]]
        points2 = [[5, 20], [80, 20], [15, 90]]
        dst = affine_transform(file, points1, points2)
        cv2.imwrite(filename + '_aff_1' + file_extension, dst)

        points1 = [[5, 5], [80, 20], [10, 80]]
        points2 = [[10, 20], [80, 20], [20, 90]]
        dst = affine_transform(file, points1, points2)
        cv2.imwrite(filename + '_aff_2' + file_extension, dst)

        points1 = np.float32([[10, 10], [100, 10], [10, 100], [100, 100]])
        points2 = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]])
        dst = perspective_transform(file, points1, points2)
        cv2.imwrite(filename + '_pers_1' + file_extension, dst)

# Start the image transformations.

# Following is an example of how to use this code.
# Create a list of image files.
imagesDir = '/Users/burton/000-Semester_06_CUNY/622_Machine_Learning/Week_10/resultdir/'
imageList = []
allFiles = [f for f in listdir(imagesDir) if isfile(join('', f))]
for file in allFiles:
    imageList.append(imagesDir + file)

# Pass the image list to transformImages function. The resulting iamges will be written to the folder from which the images were read.
#This function takes the list of image files, transforms the images and writes them to the same folder as the image.
#Please pass the full image file path to this function if running from a folder other than the folder containing the images.
transformImages(imageList)

#After running the above function you should see 10 transformations of each image in the folder containing that image.

