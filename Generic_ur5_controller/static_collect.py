import cv2
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
import time
import os
from threading import Thread
import csv

static_count = 1 #initialise data_count
dataset_size = 0 #initialise dataset_size
row_count = 0 #initialise row_count

z_depth = 0.0158#set z depth of sensor 
y_offset = -0.272 #set y offset of sensor

positions = [0.293453, 0.292474, 0.291479, 0.29049, 0.286546, 0.285603, 0.282661, 0.278635, 0.277632, 0.273683, 0.272732, 0.270595, 0.266589, 0.266617, 0.261574, 0.259516, 0.256581, 0.25459, 0.252621, 0.249597, 0.247631, 0.244634, 0.242676, 0.240696, 0.23771, 0.235728, 0.232722, 0.229721, 0.22783, 0.224832, 0.222843, 0.218854, 0.217855, 0.214886, 0.212825, 0.209831, 0.208863, 0.205882, 0.202928, 0.199938, 0.19799, 0.195935, 0.193928, 0.189936, 0.187943, 0.185935, 0.182933, 0.180924, 0.176931, 0.174945, 0.17394, 0.171935] 

for path in os.listdir('blurry'):
    # check if current path is a file
    if os.path.isfile(os.path.join('blurry', path)):
        dataset_size += 1

with (DigitSensor(serialno='D20654', resolution='QVGA', framerate='30') as digit,
            DisplayImage(window_name='DIGIT Demo') as display): #use wrapper to make accessing DIGIT sensor easier
    frame = digit.get_frame()
    print("------------Configuring brailley------------\r\n")
    brailley = kgr.kg_robot(port=30000,db_host="169.254.252.50")
    print("----------------Hi brailley!----------------\r\n\r\n")

    def read_camera(): #function to read camera
         global frame
         while True:
            frame = digit.get_frame() #get frame from camera

    def capture_frame(dir_path): #function to capture frame and save it
        os.makedirs(dir_path, exist_ok=True) 
        base_path = os.path.join(dir_path,"im{}.jpg".format(static_count)) #create path to save frame
        cv2.imwrite(base_path, frame) 
        
    def move_robot(): #fixed movements for each data collection step
        global static_count #use global count variable (need to tell function this or it doesnt work) 
        global row_count, offset
        global positions
        velocity  = 0.17
        for pos in positions:
            brailley.translatel_rel([0, 0, +0.01, 0, 0, 0], 0.5, 0.2) #move up to avoid dragging on surface
            time.sleep(0.5)
            
            brailley.movel([pos,y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move to next position
            time.sleep(0.5)

            capture_frame("sharp") #capture frame
            print("Captured frame {}".format(static_count))

            brailley.translatel_rel([0, 0, -0.01, 0, 0, 0], 0.5, 0.2)  #move down to cell
            time.sleep(0.5)
            static_count += 1 #increment counts

        scroll_button() #scroll to next row
        row_count += 1 #increment row count

    def scroll_button():
        brailley.movel([0.15, -0.261243, 0.0200194, 2.09817, 2.33561, -0.00188124], 0.5, 0.2) #move to scroll position
        time.sleep(0.5)
        brailley.translatel_rel([0, 0, -0.006, 0, 0, 0], 0.5, 0.2) #press scroll button
        time.sleep(0.5)
        brailley.translatel_rel([0, 0, 0.006, 0, 0, 0], 0.5, 0.2) #move back to scroll position
        time.sleep(0.5)
        brailley.movel([0.293484,y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move above first position
        time.sleep(0.5)
        brailley.movel([0.293484,y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move to first position

    if __name__=='__main__':
        t= Thread(target=read_camera) #start thread to read camera
        t.daemon = True #set thread to daemon so it closes when main thread closes
        t.start()

        brailley.movel([0.293484, y_offset, z_depth+0.01, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move above first position
        time.sleep(0.5)
        brailley.movel([0.293484, y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move to first position
        time.sleep(0.5)

        print("------------Starting data collection------------\r\n")
        print("------------Dataset size: {}------------\r\n".format(dataset_size))

        move_robot() #move robot to collect data

        print("------------Data collection complete------------\r\n")