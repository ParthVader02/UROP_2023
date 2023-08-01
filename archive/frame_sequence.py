import cv2 
import numpy as np
import math

frame_27 = cv2.imread("frame_27.jpg")
frame_28 = cv2.imread("frame_28.jpg")
frame_29 = cv2.imread("frame_29.jpg")

diff_1 = cv2.absdiff(frame_27, frame_28)
diff_2 = cv2.absdiff(frame_28, frame_29)
diff_3 = cv2.absdiff(frame_27, frame_29)

cv2.imshow("diff_1", diff_1)
cv2.imshow("diff_2", diff_2)
cv2.imshow("diff_3", diff_3)
cv2.waitKey(0)