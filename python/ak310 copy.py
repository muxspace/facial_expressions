import cv2
from imutils import paths
import imutils
import os
from os import listdir
from os.path import isfile
import numpy as np

def get_file_list(mypath):
	inc_extensions = ['jpeg', 'png', 'jpg']
	onlyfiles = [f for f in listdir(mypath) if any(f.endswith(ext) for ext in inc_extensions)] #read all files in mypath; uses os module
	return onlyfiles #return list of image files

def rotate(image, rows, cols):
	for i in range(0, 4):
		rotation_angle = np.random.randint(low=-60, high=60)
		M = cv2.getRotationMatrix2D((cols,rows), rotation_angle, 1)
		t = cv2.warpAffine(image, M, (cols,rows))
		if i == 0:
			img_array = t[...,np.newaxis]
		else:
			img_array = np.concatenate((img_array, t[...,np.newaxis]), axis=3)
	return img_array

def translate(image, rows, cols):
	for i in range(0, 2):
		pixel_shift = np.random.randint(low=-rows*0.2, high=rows*0.2)
		h_shift = np.float32([[1,0,pixel_shift],[0,1,0]])
		v_shift = np.float32([[1,0,0],[0,1,pixel_shift]])
		t1 = cv2.warpAffine(image, h_shift, (cols,rows))
		t2 = cv2.warpAffine(image, v_shift, (cols,rows))
		if i == 0:
			img_array = np.concatenate((t1[...,np.newaxis], t2[...,np.newaxis]), axis=3)
		else:
			img_array = np.concatenate((img_array, t1[...,np.newaxis], t2[...,np.newaxis]), axis=3)
	return img_array

def save_images(image_array, save_path, image_name):
    image_name = image_name.replace(".jpeg", "")
    for index in range(0, image_array.shape[3]):
        image = image_array[:,:,:,index]
        print(image.shape)
        cv2.imwrite(save_path+image_name+"_"+str(index)+".jpg", image)

def apply_transformations(image, save_path):
	rows, cols, dim = image.shape
	h_flip = cv2.flip(image, 1)
	v_flip = cv2.flip(image, 0)
	translated_ = translate(image, rows, cols)
	rotated_ = rotate(image, rows, cols)
	img_array = np.concatenate((h_flip[...,np.newaxis], v_flip[...,np.newaxis], translated_, rotated_), axis=3)
	return img_array

def augment_data(image_path, image_list, save_path):
    for image in image_list:
        file = image_path + image
        img0 = cv2.imread(file) #read image
        transformed_ = apply_transformations(img0, save_path)
        save_images(transformed_, save_path, image)

if __name__ == '__main__':
	image_path = 'D:\\ML\\IS622\\images\\' #location of images
	save_path = 'D:\\ML\\IS622\\images\\augmented\\'
	image_list = get_file_list(image_path)
	augment_data(image_path, image_list, save_path)
