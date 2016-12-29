library(magick)

# function receives image and writes ten transformations
# of the image into the same location as the image
ten.trans <- function(pic, pic.path, pic.name, pic.ext) {
  
  # scales image to size 200x200
  t1 <- image_scale(pic, "200x200")
  
  # adds gaussian noise to image
  t2 <- image_noise(pic, noisetype="gaussian")
  
  # adds multiplicative gaussian noise to image
  t3 <- image_noise(pic, noisetype="multiplicative gaussian")
  
  # adds impulse noise to image
  t4 <- image_noise(pic, noisetype="impulse")
  
  # adds laplacian noise to image
  t5 <- image_noise(pic, noisetype="laplacian")
  
  # adds poisson noise to image
  t6 <- image_noise(pic, noisetype="poisson")
  
  # returns upside-down mirrored image
  t7 <- image_flip(pic)
  
  # rotates image 90 degrees
  t8 <- image_rotate(pic, 90)
  
  # rotates image 314.15 degrees
  t9 <- image_rotate(pic, 314.15)
  
  # returns negative of image
  t10 <- image_negate(pic)
  
  image_write(image=t1, path=paste(pic.path, pic.name, "t1", pic.ext, sep=""))
  image_write(image=t2, path=paste(pic.path, pic.name, "t2", pic.ext, sep=""))
  image_write(image=t3, path=paste(pic.path, pic.name, "t3", pic.ext, sep=""))
  image_write(image=t4, path=paste(pic.path, pic.name, "t4", pic.ext, sep=""))
  image_write(image=t5, path=paste(pic.path, pic.name, "t5", pic.ext, sep=""))
  image_write(image=t6, path=paste(pic.path, pic.name, "t6", pic.ext, sep=""))
  image_write(image=t7, path=paste(pic.path, pic.name, "t7", pic.ext, sep=""))
  image_write(image=t8, path=paste(pic.path, pic.name, "t8", pic.ext, sep=""))
  image_write(image=t9, path=paste(pic.path, pic.name, "t9", pic.ext, sep=""))
  image_write(image=t10, path=paste(pic.path, pic.name, "t10", pic.ext, sep=""))
  
}

pic.path <- "E:/DATA 622/Week 10/"

pic.name <- "Miguel_Cotto_0001"

pic.ext <- ".jpg"

pic <- image_read(path=paste(pic.path, pic.name, pic.ext, sep=""))

ten.trans(pic, pic.path, pic.name, pic.ext)

