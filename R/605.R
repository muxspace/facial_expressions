require(imager)

##function input: a character vector of image names in the format "Rudolph_Giuliani_0022.jpg"
##function output: a character vector of image names including original images and new images
##function creates 10 additional images for each input image and saves the new images to the 
##same file location as the original images
augfunction <- function(imagesin){
  imagesnew <- character(10)
  imagesout <- imagesin
  for(i in 1:length(imagesin)){
    im <- load.image(imagesin[i])
    im.blurry <- isoblur(im,10)
    imager::save.image(im.blurry,paste("Blurry_",imagesin[i],sep=""))
    imagesnew[1] <- paste("Blurry_",imagesin[i],sep="")
    im.xedges <- deriche(im,2,order=2,axis="x")
    imager::save.image(im.xedges,paste("X_Edges_",imagesin[i],sep=""))
    imagesnew[2] <- paste("X_Edges_",imagesin[i],sep="")
    im.yedges <- deriche(im,2,order=2,axis="y")
    imager::save.image(im.yedges,paste("Y_Edges_",imagesin[i],sep=""))
    imagesnew[3] <- paste("Y_Edges_",imagesin[i],sep="")
    im.rotate90 <- imrotate(im,90)
    imager::save.image(im.rotate90,paste("Rotate_90_",imagesin[i],sep=""))
    imagesnew[4] <- paste("Rotate_90_",imagesin[i],sep="")
    im.rotate180 <- imrotate(im,180)
    imager::save.image(im.rotate180,paste("Rotate_180_",imagesin[i],sep=""))
    imagesnew[5] <- paste("Rotate_180_",imagesin[i],sep="")
    im.xmirror <- mirror(im,"x")
    imager::save.image(im.xmirror,paste("X_Mirror_",imagesin[i],sep=""))
    imagesnew[6] <- paste("X_Mirror_",imagesin[i],sep="")
    im.ymirror <- mirror(im,"y")
    imager::save.image(im.ymirror,paste("Y_Mirror_",imagesin[i],sep=""))
    imagesnew[7] <- paste("Y_Mirror_",imagesin[i],sep="")
    im.nshift <- imshift(im,100,100,boundary=1) 
    imager::save.image(im.nshift,paste("Neumann_Shift_",imagesin[i],sep=""))
    imagesnew[8] <- paste("Neumann_Shift_",imagesin[i],sep="")
    im.cshift <- imshift(im,100,100,boundary=2) 
    imager::save.image(im.cshift,paste("Circular_Shift_",imagesin[i],sep=""))
    imagesnew[9] <- paste("Circular_Shift_",imagesin[i],sep="")
    map.shift <- function(x,y) list(x=x+10,y=y+30)
    im.warp <- imwarp(im,map=map.shift)
    imager::save.image(im.warp,paste("Warp_",imagesin[i],sep=""))
    imagesnew[10] <- paste("Warp_",imagesin[i],sep="")
    imagesout <- append(imagesout, imagesnew, after = length(imagesout))
  }
    
  return(imagesout)
}
