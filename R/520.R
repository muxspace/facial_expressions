
# Required libraries
library("OpenImageR")



AugmentImage<- function(inputdir=".",outputdir="."){
    t<- NULL
    t<- data.frame('title'= list.files(inputdir))
    
    for   (i in 1:nrow(t)) {
      path <- paste(t$title[i])
      im <- readImage(gsub(" ", "",paste(inputdir,path)))
      
      im1 <- rgb_2gray(im)
      writeImage(im1, file_name = gsub(".jpg","_T01.jpg",paste(outputdir,path ))) 
      
      im2 <- flipImage(im, mode = 'horizontal')
      writeImage(im2, file_name = gsub(".jpg","_T02.jpg",paste(outputdir,path )))

      im3 <- flipImage(im, mode = 'vertical')
      writeImage(im3, file_name = gsub(".jpg","_T03.jpg",paste(outputdir,path )))

      im4 <- edge_detection(im, method = 'Frei_chen', conv_mode = 'same')
      writeImage(im4, file_name = gsub(".jpg","_T04.jpg",paste(outputdir,path )))

      im5 <- image_thresholding(im, thresh = 0.5)
      writeImage(im5, file_name = gsub(".jpg","_T05.jpg",paste(outputdir,path )))

      im6 <- gamma_correction(im, gamma = 2)
      writeImage(im6, file_name = gsub(".jpg","_T06.jpg",paste(outputdir,path )))

      im7 <- ZCAwhiten(im, k = 20, epsilon = 0.1)
      writeImage(im7, file_name = gsub(".jpg","_T07.jpg",paste(outputdir,path )))

      im8 <- delationErosion(im, Filter = c(8,8), method = 'delation')
      writeImage(im8, file_name = gsub(".jpg","_T08.jpg",paste(outputdir,path )))

      im9 <- down_sample_image(im, factor = 2.5, gaussian_blur = TRUE)
      writeImage(im9, file_name = gsub(".jpg","_T09.jpg",paste(outputdir,path )))

      im10 <- uniform_filter(im, size = c(4,4), conv_mode = 'same')
      writeImage(im10, file_name = gsub(".jpg","_T10.jpg",paste(outputdir,path )))
    }

}

