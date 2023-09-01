import os
import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

count = 0
tape = False
name = ""

if tape == False: #set name of folder to read images from
    name = "no_tape"
else:
    name = "tape"

name = ""

for path in os.listdir('/home/parth/UROP_2023/up_down_images/{}'.format(name)): #get number of images in folder
            # check if current path is a file
            if os.path.isfile(os.path.join('/home/parth/UROP_2023/up_down_images', path)):
                count += 1 # increment count

images = np.zeros((count, 320, 240)) #initialise array of images
for i in range(count):
    rgb =  cv2.imread('/home/parth/UROP_2023/up_down_images/{}/im{}.jpg'.format(name, i+1)) #read in image
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) #convert to grayscale
    images[i] = gray #add to array of images

pixel_coord = (130,120) #pixel coordinates to plot

plt.plot(images[:,pixel_coord[0],pixel_coord[1]]) #plot pixel intensity over time

import seaborn as sns; sns.set()
#ax = sns.heatmap(images.std(axis=0)) #plot standard deviation of images
plt.show()