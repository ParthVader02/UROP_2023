import cv2
import numpy as np
import os
img = cv2.imread('raw_data/im4.jpg')
b,g,r = cv2.split(img)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grey = cv2.GaussianBlur(grey, (7,7), 1.5)

dst = cv2.Laplacian(grey, cv2.CV_8U, ksize=11)

edges = cv2.Canny(dst, 10, 100)

circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 2, minDist=40,
            param1 =20 , param2=15,
            minRadius=15, maxRadius=16) #apply hough transform to detect circles in frame

if circles is not None:  #if circles are detected
    circles = np.uint16(np.around(circles)) #convert to integer array

    for i in circles[0, :]:
        centre = (i[0], i[1]) #get centre of circle
        cv2.circle(grey, centre, 1, (0, 100, 100), 3) #draw centres of circle
        cv2.circle(grey, centre, i[2], (255, 0, 255), 3) #draw circles

cv2.imshow("laplacian", dst)
cv2.imshow("canny", edges)
cv2.imshow("grey", grey)
cv2.waitKey(0)