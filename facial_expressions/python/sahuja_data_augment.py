import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

source_directory = 'Test/'
target_directory = 'Test_output/'
for file_name in os.listdir(source_directory):
    source_image =(os.path.join(source_directory, file_name))
    
    index_of_dot = file_name.index('.')
    file_name_without_extension = file_name[:index_of_dot]

    img = cv2.imread(source_image,0)
    rows,cols = img.shape
    i = 1
    # Transform 1 - Scaling - Resizing image
    dst = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    #target_image =(os.path.join(target_directory, file_name))
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transform 2 - Translation - Shift of (100,50)
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transform 3 - Rotation - Rotate 90
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transform 4 - Rotate 180 / Invert
    M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transform 5 - Erosion
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.erode(img,kernel,iterations = 1)
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transform 6 - Dilation
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.dilate(img,kernel,iterations = 1)
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transform 7 - Opening - Erosion followed by dilation
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transform 8 - Closing - Dilation followed by Erosion
    kernel = np.ones((5,5),np.uint8)
    dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transform 9 - Affine Transformation
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transform 10 - Perspective Transformation
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(300,300))
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transformation 11 - Smoothing - Blurring Image using Low-pass filter

    dst = cv2.blur(img,(5,5))
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    
    # Transformation 12 -  Change Color

    dst = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1

    # Transformation 13 - Image Filtering - 2D Convolution

    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    cv2.imwrite(target_directory+file_name_without_extension+'_'+str(i)+'.jpg', dst)
    i=i+1
