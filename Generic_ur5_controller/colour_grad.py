import cv2
import numpy as np
import os
import re

k_avg = np.zeros((128, 128))
k_list = []
for image in os.scandir('pairs/blur'):
     path = image.path
     if (path.endswith('.jpg')):
        num = (re.findall(r'\d+', path))[0]
        blur = cv2.cvtColor(cv2.imread('pairs/blur/im{}.jpg'.format(num)), cv2.COLOR_BGR2GRAY)
        sharp = cv2.cvtColor(cv2.imread('pairs/sharp/im{}.jpg'.format(num)), cv2.COLOR_BGR2GRAY)

        blur = np.array(255*blur, dtype = 'uint8')
        sharp = np.array(255*sharp, dtype = 'uint8')

        sharp_inv = np.linalg.inv(sharp)
        k = np.matmul(blur, sharp_inv)
        k_list.append(k)

k_avg = np.mean(k_list, axis=0)
raw = cv2.imread('raw_data/im2.jpg')
new_size = (128, 128) # new_size=(width, height)
raw = cv2.resize(raw, new_size)
raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
raw_blur = np.matmul(k_avg, raw)
cv2.imshow("raw", raw)
cv2.imshow("blur", raw_blur)
cv2.imshow("k", k_avg)
cv2.waitKey(0)