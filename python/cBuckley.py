#!/usr/bin/env python2
"""
Based off of compare.py
"""
import argparse
import cv2
import itertools
import os
import numpy as np
np.set_printoptions(precision=2)
import openface

HOME = '/root/openface'
IMG_DIM = 96

modelDir = os.path.join(HOME, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

align = openface.AlignDlib(
  os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(
  os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), IMG_DIM)

def create_trans(img_path, save=False):
  #this function creates 10 transformations based on the current image
  #it saves the 10 transformed images in the current directory
  img = cv2.imread(img_path)
  if img is None:
    raise Exception("Unable to load image '%s'" % img_path)
  
  rows,cols,ch = img.shape
  
  img_path_mod = img_path[:-4]
  
  aff_trans(img,cols,rows,img_path_mod)
  rot_trans(img,cols,rows,img_path_mod)
  per_trans(img,cols,rows,img_path_mod)
  
def aff_trans(img,cols,rows,img_path_mod):
    #3 affine transformations
    #transformation 1
    pts1 = np.float32([[10,10],[10,70],[70,10]])
    pts2 = np.float32([[1,1],[20,80],[80,20]])
    M1 = cv2.getAffineTransform(pts1,pts2)
    dst1 = cv2.warpAffine(img,M1,(cols,rows))
    cv2.imwrite(img_path_mod + "_at1.jpg",dst1)
    #transformation 2
    pts3 = np.float32([[1,1],[20,80],[80,20]])
    pts4 = np.float32([[10,10],[20,70],[70,20]])
    M2 = cv2.getAffineTransform(pts3,pts4)
    dst2 = cv2.warpAffine(img,M2,(cols,rows))
    cv2.imwrite(img_path_mod + "_at2.jpg",dst2)
    #transformation 3
    pts5 = np.float32([[20,20],[10,80],[80,10]])
    pts6 = np.float32([[1,1],[30,70],[70,30]])
    M3 = cv2.getAffineTransform(pts5,pts6)
    dst3 = cv2.warpAffine(img,M3,(cols,rows))
    cv2.imwrite(img_path_mod + "_at3.jpg",dst3)

def rot_trans(img,cols,rows,img_path_mod):
    #4 rotational transformations
    #transformation 1
    M4 = cv2.getRotationMatrix2D((cols/2,rows/2),75,1)
    dst4 = cv2.warpAffine(img,M4,(cols,rows))
    cv2.imwrite(img_path_mod + '_rot1.jpg',dst4)
    #transformation 2
    M5 = cv2.getRotationMatrix2D((cols/2,rows/2),150,1)
    dst5 = cv2.warpAffine(img,M5,(cols,rows))
    cv2.imwrite(img_path_mod + '_rot2.jpg',dst5)
    #transformation 3
    M6 = cv2.getRotationMatrix2D((cols/2,rows/2),225,1)
    dst6 = cv2.warpAffine(img,M6,(cols,rows))
    cv2.imwrite(img_path_mod + '_rot3.jpg',dst6)
    #transformation 4
    M7 = cv2.getRotationMatrix2D((cols/2,rows/2),300,1)
    dst7 = cv2.warpAffine(img,M7,(cols,rows))
    cv2.imwrite(img_path_mod + '_rot4.jpg',dst7)

def per_trans(img,cols,rows,img_path_mod):
    #3 rotational transformations
    #transformation 1
    pts7 = np.float32([[2,3],[93,4],[5,90],[92,91]])
    pts8 = np.float32([[0,0],[96,0],[0,96],[96,96]])
    M8 = cv2.getPerspectiveTransform(pts7,pts8)
    dst8 = cv2.warpPerspective(img,M8,(96,96))
    cv2.imwrite(img_path_mod + '_pt1.jpg',dst8)
    #transformation 2
    pts9 = np.float32([[6,7],[89,8],[9,87],[85,88]])
    pts10 = np.float32([[0,0],[96,0],[0,96],[96,96]])
    M9 = cv2.getPerspectiveTransform(pts9,pts10)
    dst9 = cv2.warpPerspective(img,M9,(96,96))
    cv2.imwrite(img_path_mod + '_pt2.jpg',dst9)
    #transformation 3
    pts11 = np.float32([[10,11],[93,12],[13,82],[83,84]])
    pts12 = np.float32([[0,0],[96,0],[0,96],[96,96]])
    M10 = cv2.getPerspectiveTransform(pts11,pts12)
    dst10 = cv2.warpPerspective(img,M10,(96,96))
    cv2.imwrite(img_path_mod + '_pt3.jpg',dst10)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  [ create_trans(p, os.path.basename(p)) for p in args.imgs ]
  pass
