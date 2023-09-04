import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt

forces = []
intensities = []
for path in os.listdir('/home/parth/UROP_2023/force_images'): #get number of images in folder
            # check if current path is a file
            if os.path.isfile(os.path.join('/home/parth/UROP_2023/force_images', path)):
                img = cv2.imread('/home/parth/UROP_2023/force_images/{}'.format(path))
                grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale
                roi = grey[140:180, 75:110] #crop image to region of interest (dot near the middle)
                #cv2.imshow("ROI", roi)
                #cv2.waitKey(0)
                intensity = np.mean(roi) #get mean pixel intensity
                intensities.append(intensity) 

                force = re.findall("\d+\.\d+", path)[0] #get force from filename
                forces.append(float(force))
forces = sorted(forces) #sort forces and intensities
intensities = sorted(intensities)
a, b = np.polyfit(forces, intensities, 1)
plt.scatter(forces, intensities, marker="x", color="red")
plt.plot(forces, a*np.array(forces) + b)
plt.xlabel("Force (N)")
plt.ylabel("Pixel intensity")
plt.show()