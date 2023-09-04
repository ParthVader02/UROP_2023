import os
import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

count = 0
tape = True
name = ""

if tape == False: #set name of folder to read images from
    name = "no_tape"
else:
    name = "tape"

for path in os.listdir('/home/parth/UROP_2023/up_down_images/{}'.format(name)): #get number of images in folder
            # check if current path is a file
            if os.path.isfile(os.path.join('/home/parth/UROP_2023/up_down_images/{}'.format(name), path)):
                count += 1 # increment count
images_tape = np.zeros((count, 320, 240)) #initialise array of images
images_no_tape = np.zeros((count, 320, 240)) #initialise array of images
mean_tape = np.zeros(count) #initialise array of mean pixel intensities
mean_no_tape = np.zeros(count) #initialise array of mean pixel intensities

for i in range(count):
    rgb_tape =  cv2.imread('/home/parth/UROP_2023/up_down_images/{}/im{}.jpg'.format(name, i+1)) #read in image
    rgb_no_tape =  cv2.imread('/home/parth/UROP_2023/up_down_images/no_tape/im{}.jpg'.format(i+1)) #read in image
    gray_tape = cv2.cvtColor(rgb_tape, cv2.COLOR_BGR2GRAY) #convert to grayscale
    gray_no_tape = cv2.cvtColor(rgb_no_tape, cv2.COLOR_BGR2GRAY) #convert to grayscale
    images_tape[i] = gray_tape #add to array of images
    images_no_tape[i] = gray_no_tape #add to array of images
    mean_tape[i] = np.mean(gray_tape[168:210, 60:90])
    mean_no_tape[i] = np.mean(gray_no_tape[168:210, 60:90])


print(mean_tape)

plt.plot(mean_tape)
plt.plot(mean_no_tape)
plt.xlabel("Image number")
plt.ylabel("Pixel intensity")
plt.legend(["Tape", "No tape"])
import seaborn as sns; sns.set()
#ax = sns.heatmap(images_tape.std(axis=0)) #plot standard deviation of images
#ax = sns.heatmap(images_no_tape.std(axis=0)) #plot standard deviation of images
plt.show()