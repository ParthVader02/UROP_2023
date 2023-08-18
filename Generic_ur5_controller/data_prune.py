import cv2
import numpy as np
import re
import os
from pynput.keyboard import Key, Listener

def process(key):
    if key == Key.tab:
        print("next image")
        return False
    if key == Key.space:
        print("delete image")
        os.remove(path)
        os.remove('blurry/im{}.jpg'.format(num)) 
        return False
         
    # by pressing 'delete' button
    # you can terminate the loop
    if key == Key.delete:
        return False
  
for image in os.scandir('sharp'): #iterate through images
    path = image.path
    num = (re.findall(r'\d+', path))[0]
    sharp = cv2.imread(path)
    blurry = cv2.imread('blurry/im{}.jpg'.format(num)) #read in image

    print(path)
    print(num)

    hori = np.concatenate((blurry, sharp), axis=1) #concatenate images
    cv2.imshow('pair', hori) #display images side by side
    cv2.waitKey(0) #wait for key press
    
    # Collect all event until released
    with Listener(on_press = process) as listener:
                listener.join()