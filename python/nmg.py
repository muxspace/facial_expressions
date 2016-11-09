#transform 500 images 10 times using opencv library
import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

mypath='/users/nathangroom/desktop/new_docker/500_photos_2'
#first transformation - rotate image 90 degrees
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i],0)
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos1/newphoto1_'+str(i)+'.jpg', dst)

#second transformation - shift image
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i],0)
    rows,cols = img.shape
    M = np.float32([[1,0,200],[0,1,90]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos2/newphoto2_'+str(i)+'.jpg', dst)
    
#third transformation -- affine transformation
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i])
    rows,cols,ch = img.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos3/newphoto3_'+str(i)+'.jpg', dst)
    
#fourth transformation -- perspective transformation
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i])
    rows,cols,ch = img.shape
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(300,300))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos4/newphoto4_'+str(i)+'.jpg', dst)
    
#fifth transformation -- rotation 190 degrees
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i],0)
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),190,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos5/newphoto5_'+str(i)+'.jpg', dst)

#sixth transformation -- combination of transformations 1 and 2
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i],0)
    rows, cols=img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    M2=M = np.float32([[1,0,100],[0,1,45]])
    dst2= cv2.warpAffine(dst,M2,(cols,rows))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos6/newphoto6_'+str(i)+'.jpg', dst2)
    
#seventh transformation -- combination of transformations 1 and 3
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i],0)
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    rows,cols=dst.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M2=cv2.getAffineTransform(pts1,pts2)
    dst2=cv2.warpAffine(dst,M2,(cols,rows))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos7/newphoto7_'+str(i)+'.jpg', dst2)
    
#eighth tranformation -- a combination of transformations 1 and 4
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i],0)
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),200,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M2 = cv2.getPerspectiveTransform(pts1,pts2)
    dst2 = cv2.warpPerspective(dst,M2,(300,300))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos8/newphoto8_'+str(i)+'.jpg', dst2)

#ninth transformation -- combination of transformations 2 and 4
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i],0)
    rows,cols = img.shape
    M = np.float32([[1,0,95],[0,1,55]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M2 = cv2.getPerspectiveTransform(pts1,pts2)
    dst2= cv2.warpPerspective(img,M2,(300,300))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos9/newphoto9_'+str(i)+'.jpg', dst2)
    
#tenth transformation--combination of transformations 3 and 4
for i in range(1,500):
    img=cv2.imread('/users/nathangroom/desktop/new_docker/500_photos_2/'+listdir(mypath)[i])
    rows,cols,ch = img.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    pts3 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts4 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M2 = cv2.getPerspectiveTransform(pts3,pts4)
    dst2= cv2.warpPerspective(dst,M2,(300,300))
    cv2.imwrite('/users/nathangroom/desktop/new_docker/new_photos10/newphoto10_'+str(i)+'.jpg', dst2)