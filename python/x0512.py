#!/usr/bin/env python
# -*- coding: utf-8 -*-

### DATA 622 Week 10 Homework
### Youqing Xiang
### user.id: x0512

"""
# Summary

# There are three main parts in this file:
#     Part I: imageProcess function
#     Part II: 10 methods used in image process
#     Part III: the main function as an example of how to run the code
   
# For imageProcess function, it starts from the legend file (image infor file),
#     and then search for the image in image file and do image transformation.
#     So, this function would ignore any image in image file if it is not or
#     not correctly recorded in legend file.
"""

import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def imageProcess(datagen,image_dir,legend_dir):
    """A function for image transformation.

    input:
        datagen: the method for image process
        image_dir: the image file direction
        legend_dir: the image infor (legend) file direction

    output:
        a dataframe: including the original image information for image process
        transformed images: saved in the new_images folder
    """
    # read legend file
    legend = pd.read_csv(legend_dir)
    image_list = legend['image']
    primary_list = legend['Primary']
    secondary_list = legend['Secondary']
    n = len(image_list)
    n = 5

    # create new_images folder for transformed images
    if not os.path.exists('new_images'):
        os.makedirs('new_images')

    # image transformation and saving original information for transformed images
    columns = ['user.id', 'image', 'Primary','Secondary']
    legend_result = pd.DataFrame(columns=columns)
    
    for i in range(0,n):
        try:
            img = load_img(os.path.join(image_dir, image_list[i]))
        except:
            print 'image reading error'
            continue
        
        df = pd.DataFrame([[legend['user.id'][i],legend['image'][i],
                            legend['Primary'][i],legend['Secondary'][i]]],
                          columns=columns)
        legend_result = legend_result.append(df)
        
        img = img_to_array(img)
        img = img.reshape((1,) + img.shape)

        name = legend['image'][i].split('.')[0]
     

        for image in datagen.flow(img,batch_size=1,save_to_dir='new_images',
                                  save_prefix=name):
            break
        
    return legend_result


# 10 methods could be used for image process
datagen1 = ImageDataGenerator(shear_range = 0.1)
  
datagen2 = ImageDataGenerator(rotation_range = 5)

datagen3 = ImageDataGenerator(width_shift_range = 0.1)
  
datagen4 = ImageDataGenerator(height_shift_range = 0.1)

datagen5 = ImageDataGenerator(shear_range = 0.1,
                              rotation_range = 5)

datagen6 = ImageDataGenerator(width_shift_range = 0.1,
                              rotation_range = 5)
  
datagen7 = ImageDataGenerator(rotation_range = 5,
                              height_shift_range = 0.1)

datagen8 = ImageDataGenerator(width_shift_range = 0.1,
                              height_shift_range = 0.1)
  
datagen9 = ImageDataGenerator(shear_range = 0.1,
                              height_shift_range = 0.1)

datagen10 = ImageDataGenerator(shear_range = 0.1,
                               width_shift_range = 0.1)

datagens = [datagen1,datagen2,datagen3,datagen4,datagen5,
              datagen6,datagen7,datagen8,datagen9,datagen10]



if __name__ == "__main__":
    image_dir = 'Xiang_image'
    legend_dir = 'Xiang_data/legend.csv'

    columns = ['user.id', 'image', 'Primary','Secondary']
    legend = pd.DataFrame(columns=columns)

    for datagen in datagens:
        df = imageProcess(datagen,image_dir,legend_dir)
        legend = pd.concat([legend,df])
    
    newlist = pd.Series(os.listdir('new_images'))
    legend = legend.sort_values(by='image')
    legend.reset_index(drop=True,inplace=True)
    
    legend['image'] = newlist

    if not os.path.exists('new_data'):
        os.makedirs('new_data')
    
    legend.to_csv('new_data/new_legend.csv')
