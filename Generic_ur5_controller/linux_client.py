import cv2
import numpy as np
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
import pandas as pd
import time
import matplotlib.pyplot as plt
import socket
from PIL import Image  
import PIL  
import random

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server
data_count = 1 #initialise data_count

import os, glob
for filename in glob.glob("frame_*"):
    os.remove(filename) 

with (DigitSensor(serialno='D20654', resolution='QVGA', framerate='60') as digit,
            DisplayImage(window_name='DIGIT Demo') as display): #use wrapper to make accessing DIGIT sensor easier
    
    print("------------Configuring brailley------------\r\n")
    brailley = kgr.kg_robot(port=30000,db_host="169.254.252.50")
    print("----------------Hi brailley!----------------\r\n\r\n")

    def capture_frame(dir_path): #function to capture frame and save it
        os.makedirs(dir_path, exist_ok=True) 
        frame = digit.get_frame() 
        base_path = os.path.join(dir_path,"im{}.jpg".format(data_count)) #create path to save frame
        cv2.imwrite(base_path, frame) 

    def move_robot(): #fixed movements for each data collection step
        brailley.movel([0.260394, -0.264857, 0.0135985, 2.09924, 2.33714, -0.000203997], 0.5, 0.2) #move to first position
        time.sleep(1)

        for i in range(0, 10): 
            brailley.translatel_rel([random.uniform(0.00, 0.00),random.uniform(0.00, 0.00),random.uniform(0.00, 0.00),0,0,0], acc=0.05, vel=0.1, wait=True) 
            time.sleep(0.5) #move to random positions [find the ranges experimentally]
            capture_frame("home/parth/UROP_2023/raw_data") #capture frame
            count += 1 #increment counts
        brailley.movel([0.260394, -0.264857, 0.0135985, 2.09924, 2.33714, -0.000203997], 0.5, 0.2) 

    if __name__=='__main__':
        print("------------Starting data collection------------\r\n")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            while data_count < 500: #set total number of data points to collect
                move_robot() #move robot to random positions and capture frames
                s.sendall(b'change') #send message to server to change word
                data = s.recv(1024)
                if data == b'changeconfirm': #if server confirms change
                    print("------------Changing word------------\r\n")
            s.sendall(b'close') #close connection
            print("------------Data collection complete------------\r\n")