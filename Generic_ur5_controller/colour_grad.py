import cv2
import numpy as np
import os
img = cv2.imread('raw_data/im5.jpg')
b,g,r = cv2.split(img)

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)

cv2.waitKey(0)