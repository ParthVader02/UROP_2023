import cv2
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
import time
import os
from threading import Thread
import csv
import random

dataset_size = 28
dynamic_count = 1

velocity = 0.9 #set velocity for sliding motion
start = time.time()
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
            frame = digit.get_frame() #get frame from camera
            if slide_capture_flag == True:
                capture_frame("blurry")
                elapsed = time.time() - start
                time_list.append(elapsed)

    def capture_frame(dir_path): #function to capture frame and save it
        global static_collect, dynamic_collect, static_count, dynamic_count
        os.makedirs(dir_path, exist_ok=True)  
        base_path = os.path.join(dir_path,"im{}.jpg".format(dynamic_count)) #create path to save frame
        cv2.imwrite(base_path, frame)
        dynamic_count += 1 #increment counts
    
    def move_robot(): #fixed movements for each data collection step
        global dynamic_count, velocity
        global slide_capture_flag
        global start, time_list

        time_list = [] #reset time list
        start = time.time()
        slide_capture_flag = True
        brailley.movel([0.17, -0.271902, 0.0172491, 2.09818, 2.33554, -0.00188674], 500, velocity) #slide across one row
        slide_capture_flag = False
        time.sleep(0.5)
        scroll_button()
        with open('time_list.csv', 'a') as f:
                write = csv.writer(f)
                time_list.insert(0, velocity)
                write.writerow(time_list)

    def scroll_button():
        brailley.movel([0.155901, -0.261243, 0.0200194, 2.09817, 2.33561, -0.00188124], 0.5, 0.2) #move to scroll position
        time.sleep(0.5)
        brailley.translatel_rel([0, 0, -0.004, 0, 0, 0], 0.5, 0.2) #press scroll button
        time.sleep(0.5)
        brailley.translatel_rel([0, 0, 0.004, 0, 0, 0], 0.5, 0.2) #move back to scroll position
        time.sleep(0.5)
        brailley.movel([0.290128, -0.271902, 0.02, 2.09818, 2.33554, -0.00188674], 0.5, 0.2) #move above first position
        time.sleep(0.5)
        brailley.movel([0.290128, -0.271902, 0.0172491, 2.09818, 2.33554, -0.00188674], 0.5, 0.2) #move to first position

    if __name__=='__main__':
        t= Thread(target=read_camera) #start thread to read camera
        t.daemon = True #set thread to daemon so it closes when main thread closes
        t.start()
        
        brailley.movel([0.290128, -0.271902, 0.02, 2.09818, 2.33554, -0.00188674], 0.5, 0.2) #move above first position
        time.sleep(0.5)
        brailley.movel([0.290128, -0.271902, 0.0172491, 2.09818, 2.33554, -0.00188674], 0.5, 0.2) #move to first position
        time.sleep(0.5)

        with open('time_list.csv', 'w') as f:
                write = csv.writer(f)

        print("------------Starting data collection------------\r\n")

        while dynamic_count < dataset_size:
            velocity = random.uniform(0.2, 0.8) #randomise velocity
            move_robot() #movements
            print("Data point {} of {} collected".format(dynamic_count, dataset_size)) #print progress

print("------------Data collection complete------------\r\n")