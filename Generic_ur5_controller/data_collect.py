import cv2
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
import time
import random
import os

data_count = 1 #initialise data_count

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
        capture_frame("raw_data") #capture frame
        data_count += 1 #increment counts
        brailley.translatel_rel([-0.0058, 0, 0, 0, 0, 0], 0.5, 0.2) #move to next position
        brailley.translatel_rel([random.uniform(-0.002, 0.002),random.uniform(-0.002, 0.002),random.uniform(-0.002, 0.002),0,0,0], acc=0.05, vel=0.1, wait=True) 

    if __name__=='__main__':
        brailley.movel([0.260394, -0.264857, 0.0135985, 2.09924, 2.33714, -0.000203997], 0.5, 0.2) #move to first position
        time.sleep(2) #wait for camera colours to adjust

        print("------------Starting data collection------------\r\n")
        dataset_size = 500 #set total number of data points to collect

        while data_count < dataset_size: 
            if data_count%20 == 0: #every 20 data points, scroll
                brailley.movel([0.260394, -0.264857, 0.0135985, 2.09924, 2.33714, -0.000203997], 0.5, 0.2) #move to scroll position
                time.sleep(0.5)
                brailley.translatel_rel([0, 0, -0.004, 0, 0, 0], 0.5, 0.2) #press scroll button
                time.sleep(0.5)
                brailley.translatel_rel([0, 0, -0.004, 0, 0, 0], 0.5, 0.2) #move back to scroll position
                time.sleep(0.5)
                brailley.movel([0.260394, -0.264857, 0.0135985, 2.09924, 2.33714, -0.000203997], 0.5, 0.2) #move to first position
            else:
                move_robot() #move robot to next position and capture frame
        print("------------Data collection complete------------\r\n")