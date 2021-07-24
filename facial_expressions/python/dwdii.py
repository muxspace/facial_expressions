#
# Author: Daniel Dittenhafer
#
#     Created: Nov 3, 2016
#
# Description: Image Transformations for faces
#
# An example of each transformation can be found here: https://github.com/dwdii/emotional-faces/tree/master/data/transformed
#
__author__ = 'Daniel Dittenhafer'
import os
import numpy as np
from scipy import misc
from scipy import ndimage
import cv2

def saveImg(destinationPath, prefix, filepath, imgData):
    """Helper function to enable a common way of saving the transformed images."""
    fileName = os.path.basename(filepath)
    destFile = destinationPath + "\\" + prefix + "-" + fileName
    misc.imsave(destFile, imgData)

def reflectY(img):

    tx = [[1, 0], [0, -1]]
    offset = [0, 350]
    img2 = ndimage.interpolation.affine_transform(img, tx, offset)

    return img2

def rotate5(img):

    img2 = cv2.resize(img, (385, 385), interpolation=cv2.INTER_AREA)

    # Rotate
    a = 5.0 * np.pi / 180.0
    tx = [[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]

    offset = [-10,25]  # [right, down] negatives go other direction
    img2 = ndimage.interpolation.affine_transform(img2, tx, offset)

    # Zoom
    img2 = img2[10:360, 10:360]

    return img2


def cvErode(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/3/ch03lvl1sec32/Cartoonizing+an+image"""
    kernel = np.ones((5, 5), np.uint8)

    img_erosion = cv2.erode(img, kernel, iterations=1)


    return img_erosion


def cvDilate(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/3/ch03lvl1sec32/Cartoonizing+an+image"""
    kernel = np.ones((5, 5), np.uint8)

    img_dilation = cv2.dilate(img, kernel, iterations=1)

    return img_dilation

def cvDilate2(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/3/ch03lvl1sec32/Cartoonizing+an+image"""
    kernel = np.ones((5, 5), np.uint8)

    img_dilation = cv2.dilate(img, kernel, iterations=2)

    return img_dilation

def cvMedianBlur(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/3/ch03lvl1sec32/Cartoonizing+an+image"""

    img2 = cv2.medianBlur(img, 7 )

    return img2


def cvExcessiveSharpening(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec22/Sharpening"""
    kernel_sharpen_1 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    img2 = cv2.filter2D(img, -1, kernel_sharpen_1)
    return img2

def cvEdgeEnhancement(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec22/Sharpening"""
    kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0

    img2 = cv2.filter2D(img, -1, kernel_sharpen_3)
    return img2

def cvBlurMotion1(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec23/Embossing"""
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    img2 = cv2.filter2D(img, -1, kernel_motion_blur)
    return img2

def cvBlurMotion2(img):
    """https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec23/Embossing"""
    size = 30
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    img2 = cv2.filter2D(img, -1, kernel_motion_blur)
    return img2