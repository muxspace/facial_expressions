# Augmenting Face Image Database via Transformations (Flip, Rotate, Crop & Scale)
# Jay Narhan
# UserID: JN721
#

# This class is designed to apply a series of image transformations to a set of images. A transformation is simply a
# function. It is a function that maps the image to another version of the image.
#
# G(x,y) = T( f(x,y) )
#
# G will be the new image based on the Transformation T. For every image, f(x,y) there will be 10 transformations
# generated.
#
# Images (including the original) may be saved to disk, along with a reference file that tracks the labelled emotion for
# each original image. Saving to disk requires invocation of the script with the argument "-s" at the cmd line.

# As a Class of type NN_Images, the object can be used for in-memory processing as required.
# Usage: Import this Class via "from NN_images import *" and call the methods needed.

import math, os, sys
import numpy as np
import pandas as pd

from skimage.io import imread, imsave
from skimage import transform as tf
from skimage import img_as_float


class NN_Images(object):
    def __init__(self):
        self.images       = dict()

        self.ROOT_DIR     = os.getcwd()
        self.IMAGE_DIR    = './images'
        self.DATA_DIR     = './data'
        self.TRANS_DIR    = './trans_res'

        self.IMG_WIDTH    = 350
        self.IMG_HEIGHT   = 350

        self.legends_file = 'legend.csv'

    def get_imgs(self):
        return self.images

    def transform_img__(self, img, fn, emotion):
        self.images[fn] = {'Image': img, 'Emotion': emotion}                                            # Store original
        counter = 0

        self.images["Trans" + str(counter) + "_" + fn] = {'Image': np.fliplr(img), 'Emotion': emotion}  # FLIP the image
        counter += 1

        for deg in range(-10, 15, 5):                                        # ROTATE to be robust to camera orientation
            if deg == 0:
                continue

            self.images["Trans" + str(counter) + "_" + fn] = {'Image': tf.rotate(img, deg), 'Emotion': emotion}
            counter += 1

        lenX, lenY = img.shape                                                           # CROP based on rough heuristic
        for crop_size in range(8, 14, 2):
            cropped = img[lenX / crop_size: - lenX / crop_size, lenY / crop_size: - lenY / crop_size]
            self.images["Trans" + str(counter) + "_" + fn] = {'Image': cropped, 'Emotion': emotion}
            counter += 1

        for i in range(2):                                           # SCALE down images (random factor btw 1.1 to 1.21)
            scale_factor = math.sqrt(1.1) ** np.random.randint(2, 5)
            scaled_img = tf.warp(img, tf.AffineTransform(scale=(scale_factor, scale_factor)))
            self.images["Trans" + str(counter) + "_" + fn] = {'Image': scaled_img, 'Emotion': emotion}
            counter += 1

    def process_imgs(self):
        # Read the file that tracks the emotions against the original images. Each new transformed image, will carry the
        # same emotion label.
        try:
            os.chdir(self.DATA_DIR)
            legend = pd.read_csv(self.legends_file)

        except IOError as e:
            print "I/O Error ({0}).".format(e.args[0])
            sys.exit(2)

        except OSError as e:
            print "O/S Error({0}:{1})".format(e.args[1], self.DATA_DIR)
            sys.exit(2)

        finally:
            os.chdir(self.ROOT_DIR)

        os.chdir(self.IMAGE_DIR)

        processed_imgs = 0
        for filename in os.listdir(os.getcwd()):
            try:
                img = img_as_float(imread(filename))  # Read file as a float

                # Pre-process:
                rows, cols = img.shape
                if cols != self.IMG_WIDTH or rows != self.IMG_HEIGHT:
                    print 'Resizing image ... '
                    img = tf.resize(img, output_shape=(self.IMG_WIDTH, self.IMG_HEIGHT))

                emotion = legend.loc[legend['image'] == filename, 'emotion'].iloc[0]     # Track the emotion of original
                self.transform_img__(img, filename, emotion)
                processed_imgs += 1

            except IOError as e:
                print "WARNING: {0} ... skipping this non-image file.".format(e.args[0])

        print 'Processed {0} images'.format(processed_imgs)

        os.chdir(self.ROOT_DIR)

    def S2D(self, userid):

        os.chdir(self.ROOT_DIR)

        if not os.path.exists(self.TRANS_DIR):
            os.makedirs('trans_res')

        os.chdir(self.TRANS_DIR)

        legend = pd.DataFrame(columns=['user.id', 'image', 'emotion'])

        try:
            for name, data in self.images.iteritems():
                imsave(name, data['Image'])                                                         # Save image to disk
                df = pd.DataFrame([[userid,
                                    name, data['Emotion']]],
                                  columns=['user.id', 'image', 'emotion'])
                legend = legend.append(df)

            legend = legend.sort_values(by='image')
            legend.to_csv('01_legend.csv', index=False)                                   # More efficient write to disk

        except:
            print 'Unknown Error in Saving to Disk'
            pass

        finally:
            os.chdir(self.ROOT_DIR)
