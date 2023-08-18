import cv2
import numpy as np
import os
import re
from scipy.spatial import KDTree, cKDTree
import matplotlib.pyplot as plt

img = cv2.imread("sharp/im5.jpg")

thresh = 0.01

sift = cv2.SIFT_create(contrastThreshold=thresh, edgeThreshold=10, sigma=0.8)
#contrastThreshold basically controls the number of keypoints detected by strength
#edgeThreshold doesnt have much effect so kept at default
#sigma controls gaussian blur, reduced from default because camera not very strong, keep this constant
kp = sift.detect(img,None)
pts =  [key_point.pt for key_point in kp]
pts = np.array(pts)

print(pts)

tree = KDTree(pts)

for i in range(len(pts)):
    ds, inds =  tree.query(pts[i], 4) #get 3 nearest neighbours
    #print(ds)
    for j in range(1,len(inds)):
        if ds[j] < 1: #remove points that are too close
            new_pt  = (pts[i] + pts[inds[j]])/2
            #print(new_pt)
            pts[i] = new_pt
            np.delete(pts, inds[j])
    cv2.circle(img, (int(pts[i][0]),int(pts[i][1])), 1, (0,0,255), -1)  
#print(pts)

#cv2.circle(img, (int(pts[inds[1]][0]),int(pts[inds[1]][1])), 3, (0,255,0), -1)    
cv2.imshow("img", img)
cv2.waitKey(0)