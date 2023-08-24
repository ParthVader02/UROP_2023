import cv2
import numpy as np
import os
import re
for image in os.scandir('test_blurry'):
    path = image.path
    num = (re.findall(r'\d+', path))[0]

    img = cv2.imread(path)
    new_size = (128, 128) # new_size=(width, height)
    img = cv2.resize(img, new_size)

    os.makedirs('test', exist_ok=True) 
    base_path = os.path.join('test',"im{}.jpg".format(num)) #training inputs
    cv2.imwrite(base_path, img)