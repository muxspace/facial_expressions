from keras.preprocessing.image import ImageDataGenerator
import glob
import matplotlib.pyplot as pyplot
from PIL import Image
import numpy as np
import os
import pandas as pd


#################################################################################

## To run this code below are the things to be done

## 1. Define the "image_dir_in" path and keep the input images there  (509 in this case )
## 1. Define the "image_dir_out" path where the augmented images will be saved
## 2. Define the "legend_file_in"  path & input legend file where from the name of the images along with the emotions will be read
## 2. Define the "legend_file_out"  path where the name of the images along with the emotions will be written

#################################################################################


image_dir_in=r"C:\Users\sandnand\Google Drive\facial_expressions\images\*.jpg"
legend_file_in=r"C:\Users\sandnand\Google Drive\facial_expressions\data\legend.csv"
legend_file_out=r"C:\Users\sandnand\Google Drive\facial_expressions\data\legend_out.csv"
image_dir_out=r"C:\Users\sandnand\Google Drive\facial_expressions\Augmented"




image_names=glob.glob(image_dir_in,recursive=False)
file=pd.read_csv(legend_file_in)
n_images=len(image_names)

X_image=[]
y_image=[]

emotions=['neutral', 'anger', 'surprise', 'sadness', 'happiness', 'contempt']
i_emotions=[1,2,3,4,5,6]
d_emotions1=dict(zip(emotions,i_emotions))
d_emotions2=dict(zip(i_emotions,emotions))



for i in range(n_images):
    im=Image.open(image_names[i])
    sz=im.size
    im_arr= list(im.getdata())
    im_matrix=np.array(im_arr).reshape(sz)
    X_image.append(im_matrix)

    im_nm = os.path.basename(image_names[i])
    im_emotion=file[file['image'] == im_nm]['emotion'].values
    y_image.append( d_emotions1[im_emotion[0]] )

X_image=np.reshape(X_image,(509,1,350,350))
X_image=X_image.astype('float32')

y_image=np.reshape(y_image,(509,1,1))



datagen_fn = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen_fn.fit(X_image)


datagen_rr = ImageDataGenerator(rotation_range=90)
datagen_rr.fit(X_image)


datagen_zoom = ImageDataGenerator(zoom_range=0.5)  ###   range becomes a random from 50% to 150%
datagen_zoom.fit(X_image)

shift = 0.2
datagen_rs = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)  ### the range is a random number in between 1-0.2 to 1+0.2
datagen_rs.fit(X_image)


datagen_fp = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
datagen_fp.fit(X_image)

bs=509

dict={}

##  datagen.flow generates a continuous  flow based on batch size

k=0
for j in range(0,2):


    for X_batch, y_batch in datagen_fn.flow(X_image, y_image, batch_size=bs):
        for i in range(0, bs):
            augfile= image_dir_out + '\Augmented_FN_'  + 'batch_'+ str(j) + '_file_'+ str(i) + '.jpg'
            baseaugfile=os.path.basename(augfile)
            pyplot.imsave( augfile,X_batch[i].reshape(350, 350), cmap=pyplot.get_cmap('gray'))
            dict[k]={'user.id':1122,'image':baseaugfile,'emotion': d_emotions2[y_batch[i][0][0]]}
            k+=1

        break


    for X_batch, y_batch in datagen_rr.flow(X_image, y_image, batch_size=bs):
        for i in range(0, bs):
            augfile=image_dir_out + '\Augmented_RR_' + 'batch_'+ str(j) + '_file_'+ str(i) + '.jpg'
            baseaugfile=os.path.basename(augfile)
            pyplot.imsave( augfile,X_batch[i].reshape(350, 350), cmap=pyplot.get_cmap('gray'))
            dict[k]={'user.id':1122,'image':baseaugfile,'emotion': d_emotions2[y_batch[i][0][0]]  }
            k+=1

        break

    for X_batch, y_batch in datagen_zoom.flow(X_image, y_image, batch_size=bs):
        for i in range(0, bs):
            augfile = image_dir_out + '\Augmented_ZM_'  + 'batch_' + str(j) + '_file_' + str(i) + '.jpg'
            baseaugfile = os.path.basename(augfile)
            pyplot.imsave(augfile, X_batch[i].reshape(350, 350), cmap=pyplot.get_cmap('gray'))
            dict[k] = {'user.id': 1122, 'image': baseaugfile, 'emotion': d_emotions2[y_batch[i][0][0]]}
            k += 1

        break

    for X_batch, y_batch in datagen_rs.flow(X_image, y_image, batch_size=bs):
        for i in range(0, bs):
            augfile = image_dir_out + '\Augmented_RS_' + 'batch_' + str(
                j) + '_file_' + str(i) + '.jpg'
            baseaugfile = os.path.basename(augfile)
            pyplot.imsave(augfile, X_batch[i].reshape(350, 350), cmap=pyplot.get_cmap('gray'))
            dict[k] = {'user.id': 1122, 'image': baseaugfile, 'emotion': d_emotions2[y_batch[i][0][0]]}
            k += 1

        break

    for X_batch, y_batch in datagen_fp.flow(X_image, y_image, batch_size=bs):
        for i in range(0, bs):
            augfile = image_dir_out + '\Augmented_FP_' + 'batch_' + str(j) + '_file_' + str(i) + '.jpg'
            baseaugfile = os.path.basename(augfile)
            pyplot.imsave(augfile, X_batch[i].reshape(350, 350), cmap=pyplot.get_cmap('gray'))
            dict[k] = {'user.id': 1122, 'image': baseaugfile, 'emotion': d_emotions2[y_batch[i][0][0]]}
            k += 1

        break


df_vals=[(v) for (k,v) in dict.items()]
df=pd.DataFrame(df_vals)

df.to_csv(legend_file_out)


