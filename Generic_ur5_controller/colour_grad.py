import cv2
import numpy as np
import os
import re

cv2.imshow("raw", raw)
cv2.imshow("blur", raw_blur)
cv2.imshow("k", k_avg)
cv2.waitKey(0)