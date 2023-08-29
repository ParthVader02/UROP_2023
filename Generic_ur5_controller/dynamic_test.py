import cv2
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
import time
import os
from threading import Thread
import csv
import random

dataset_size = 20 #set approx dataset size
dynamic_count = 1
row_counter = 1

z_depth = 0.0143 #set z depth of sensor, with medical tape need to be lower for clarity
y_offset = -0.27 #set y offset of sensor

velocity = 0 #initialise velocity
start = 0
time_list = []
slide_capture_flag = False #initialise slide capture flag

with (DigitSensor(serialno='D20654', resolution='QVGA', framerate='60') as digit,
            DisplayImage(window_name='DIGIT Demo') as display): #use wrapper to make accessing DIGIT sensor easier
    frame = digit.get_frame()
    print("------------Configuring brailley------------\r\n")
    brailley = kgr.kg_robot(port=30000,db_host="169.254.252.50")
    print("----------------Hi brailley!----------------\r\n\r\n")

    def read_camera(): #function to read camera
         global frame
         global start, time_list
         while True:
            t = time.time()
            start = t -start 
            frame = digit.get_frame() #get frame from camera
            time_of_capture =t #average time of capture
            if slide_capture_flag == True: 
                capture_frame("test_blurry") #capture frame
                time_list.append(time_of_capture)

    def capture_frame(dir_path): #function to capture frame and save it
        global static_collect, dynamic_collect, static_count, dynamic_count
        os.makedirs(dir_path, exist_ok=True)  
        base_path = os.path.join(dir_path,"im{}.jpg".format(dynamic_count)) #create path to save frame
        cv2.imwrite(base_path, frame)
        dynamic_count += 1 #increment counts
    
    def move_robot(): #fixed movements for each data collection step
        global dynamic_count, velocity, row_counter
        global slide_capture_flag
        global start, time_list

        time_list = [] #reset time list
        slide_capture_flag = True
        start_1= time.time()
        brailley.movel([0.293484, y_offset, z_depth,  2.21745, 2.22263, -0.00201733], 500, velocity) #slide across one row
        slide_capture_flag = False
        end = time.time()
        print(end-start_1)
        time.sleep(0.5)
        scroll_button() #press scroll button
        with open('test_time_list.csv', 'a') as f: #append string of letters, velocity and times to csv
                write = csv.writer(f)
                time_list.insert(0, velocity) #insert velocity at start of list
                with open('training.txt', 'r') as f:
                    doc = f.read()
                    row_string = doc[(20*row_counter):(20*row_counter) +20] #get string of letters for each row
                    time_list.insert(0, row_string) #insert string of letters at start of list
                    row_counter += 1
                write.writerow(time_list) #write row to csv

    def scroll_button():
        brailley.movel([0.15, -0.261243, 0.0200194, 2.09817, 2.33561, -0.00188124], 0.5, 0.2) #move to scroll position
        time.sleep(0.5)
        brailley.translatel_rel([0, 0, -0.006, 0, 0, 0], 0.5, 0.2) #press scroll button
        time.sleep(0.5)
        brailley.translatel_rel([0, 0, 0.006, 0, 0, 0], 0.5, 0.2) #move back to scroll position
        time.sleep(0.5)
        brailley.movel([0.293484,y_offset, z_depth+0.01, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move above first position
        time.sleep(0.5)
        brailley.movel([0.293484,y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move to first position

    if __name__=='__main__':
        t= Thread(target=read_camera) #start thread to read camera
        t.daemon = True #set thread to daemon so it closes when main thread closes
        t.start()
        
        brailley.movel([0.169, y_offset, z_depth+0.01, 2.21745, 2.22263, -0.00101733], 0.5, 0.2) #move above first position
        time.sleep(0.5)
        brailley.movel([0.169, y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move to first position
        time.sleep(0.5)

        with open('test_time_list.csv', 'w') as f:
                write = csv.writer(f) #create csv file to write to

        print("------------Starting data collection------------\r\n")

        while dynamic_count < dataset_size: #get at least the target data set size
            velocity = 0.2 #set velocity
            move_robot() #movements
            print("Data point {} of {} collected".format(dynamic_count, dataset_size)) #print progress

print("------------Data collection complete------------\r\n")