import cv2
import numpy as np
import os
import re

img = cv2.imread("sharp/im35.jpg")

thresholds = [0.02, 0.03]
features = []

for thresh in thresholds:
    sift = cv2.SIFT_create(contrastThreshold=thresh, edgeThreshold=10, sigma=0.8)
    #contrastThreshold basically controls the number of keypoints detected by strength
    #edgeThreshold doesnt have much effect so kept at default
    #sigma controls gaussian blur, reduced from default because camera not very strong, keep this constant
    kp = sift.detect(img,None)
    pts =  [key_point.pt for key_point in kp]
    features.append(pts)


print(features)
for pts in features:
    for x,y in pts:
        cv2.circle(img, (int(x),int(y)), 1, (0,0,255), -1)      

cv2.imshow('img',img)


cv2.waitKey(0)