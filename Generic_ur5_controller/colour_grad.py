import cv2
import numpy as np
img = cv2.imread('raw_data/im1.jpg')
b,g,r = cv2.split(img)
#r = cv2.GaussianBlur(r, (5,5),0)
#g = cv2.GaussianBlur(g, (5,5),0)
#b = cv2.GaussianBlur(b, (5,5),0)

# define the alpha and beta
alpha = 1.5 # Contrast control
beta = 10 # Brightness control

# call convertScaleAbs function
adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# display the output image
cv2.imshow('adjusted', adjusted)

#get x and y gradeints
gx = cv2.Sobel(r,  cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
gy = cv2.Sobel(r,  cv2.CV_32F ,0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
cv2.imshow('grad_x', gx)
cv2.imshow('grad_y', gy)

magnitude = np.sqrt((gx ** 2) + (gy ** 2))
orientation = np.arctan2(gx, gx) * (180 / np.pi) % 180
cv2.imshow('magnitude', magnitude)


cv2.imshow('original', img)
cv2.imshow('red', r)
cv2.imshow('green', g)
cv2.imshow('blue', b)

cv2.waitKey(0)