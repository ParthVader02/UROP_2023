# loading library
import cv2
import numpy as np
import os
from skimage.util import random_noise
import random
import re

folder_dir = 'raw_data'
count = 1

for image in os.scandir(folder_dir):
    path = image.path
    if (path.endswith('.jpg')):
        num = (re.findall(r'\d+', path))[0]
        img = cv2.imread(path)
        kernel_size = random.randint(25,75) #change this to change blur amount -> between 25 and 75 is decent
        kernel_h = np.zeros((kernel_size, kernel_size)) #create empty kernel
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) #create horizontal kernel
        kernel_h /= kernel_size #normalise kernel

        horizonal_mb = cv2.filter2D(img, -1, kernel_h)
        noise_img = random_noise(horizonal_mb, mode='gaussian', var=0.02**2) #gives good approx to actual frames
        noise_img = np.array(255*noise_img, dtype = 'uint8')

        os.makedirs('noisy_data', exist_ok=True) 
        base_path = os.path.join('noisy_data',"im{}.jpg".format(num)) #create path to save frame
        cv2.imwrite(base_path, noise_img)
        count +=1
print(count)