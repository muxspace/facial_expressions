require(jpeg)
require(OpenImageR)
require(squash)


augementImages <- function(inputdir = ".", outputdir = "."){
  fileList <- list.files(path = inputdir)
  
  lapply(fileList, function(X){
    filePath <- paste0(inputdir, X)
    img <- readJPEG(filePath)
    filename <- strsplit(X, ".", fixed=TRUE)[[1]][1]
    
    
    res = Augmentation(img, resiz_width = 50, resiz_height = 200, rotate_angle = 90)
    savemat(res, paste0(outputdir, filename,"_1",".jpg"))
    
    res = Augmentation(img, resiz_width = 75, resiz_height = 100, rotate_angle = 90)
    savemat(res, paste0(outputdir, filename,"_2",".jpg"))
    
    res = Augmentation(img, resiz_width = 500, resiz_height = 150, rotate_angle = 90)
    savemat(res, paste0(outputdir, filename,"_3",".jpg"))
    
    res = Augmentation(img, resiz_width = 150, resiz_height = 350, rotate_angle = 90)
    savemat(res, paste0(outputdir, filename,"_4",".jpg"))
    
    res = Augmentation(img, resiz_width = 100, resiz_height = 100, rotate_angle = 180)
    savemat(res, paste0(outputdir, filename,"_5",".jpg"))
    
    res = Augmentation(img, resiz_width = 175, resiz_height = 50, rotate_angle = 180)
    savemat(res, paste0(outputdir, filename,"_6",".jpg"))
    
    res = Augmentation(img, resiz_width = 50, resiz_height = 175, rotate_angle = 180)
    savemat(res, paste0(outputdir, filename,"_7",".jpg"))
    
    res = Augmentation(img, resiz_width = 275, resiz_height = 75, rotate_angle = 180)
    savemat(res, paste0(outputdir, filename,"_8",".jpg"))
    
    res = Augmentation(img, resiz_width = 75, resiz_height = 75, rotate_angle = 45)
    savemat(res, paste0(outputdir, filename,"_9",".jpg"))
    
    res = Augmentation(img, resiz_width = 175, resiz_height = 75, rotate_angle = 45)
    savemat(res, paste0(outputdir, filename,"_10",".jpg"))
  })
  
}

## Usage augementImages()

## Fetches the list of files from the input directory and outputs the augmentated images into the output directory


## inputdir : The path to the input directory from where the images will be fetched
## output dir : The path to the output directory where the augmented images will be stored. 
