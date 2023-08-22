import cv2
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
import time
import os
from threading import Thread
import csv
from pynput.keyboard import Key, Listener

static_count = 1 #initialise data_count
dataset_size = 0 #initialise dataset_size
row_count = 0 #initialise row_count

z_depth = 0.016#set z depth of sensor 
y_offset = -0.271 #set y offset of sensor
positions = [] #initialise list of positions

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
            cv2.imshow('frame', frame) #display frame
            cv2.waitKey(1) #wait for key press

    def capture_frame(dir_path): #function to capture frame and save it
        os.makedirs(dir_path, exist_ok=True) 
        base_path = os.path.join(dir_path,"im{}.jpg".format(static_count)) #create path to save frame
        cv2.imwrite(base_path, frame) 

    def process(key):
        global positions
        dx = 0
        if key == Key.tab:
            print("left")
            dx = -0.001
        if key == Key.space:
            print("right")
            dx = 0.001
        if key == Key.enter:
            print("position saved")
            print(brailley.getl()[0])
            #capture_frame("sharp") #capture frame
            positions.append(brailley.getl()[0]) #append position to list
        if key == Key.delete:
            with open('positions.csv', 'w') as f:
                write = csv.writer(f)
                write.writerow(positions)
            return False
        
        brailley.translatel_rel([dx,0,0,0,0,0], 0.5, 0.2) #move to next position
        time.sleep(0.5)
        return False
         

    def move_robot(): #fixed movements for each data collection step
        global static_count #use global count variable (need to tell function this or it doesnt work) 
        global row_count, offset
        global length
        velocity  = 0.12
        while True:
            # Collect all event until released
            with Listener(on_press = process) as listener:
                        listener.join()
            #static_count += 1 #increment counts

            #scroll_button() #scroll to next row
            #row_count += 1 #increment row count

    def scroll_button():
        brailley.movel([0.15, -0.261243, 0.0200194, 2.09817, 2.33561, -0.00188124], 0.5, 0.2) #move to scroll position
        time.sleep(0.5)
        brailley.translatel_rel([0, 0, -0.006, 0, 0, 0], 0.5, 0.2) #press scroll button
        time.sleep(0.5)
        brailley.translatel_rel([0, 0, 0.006, 0, 0, 0], 0.5, 0.2) #move back to scroll position
        time.sleep(0.5)
        brailley.movel([0.172,y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move above first position
        time.sleep(0.5)
        brailley.movel([0.172,y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move to first position

    if __name__=='__main__':
        t= Thread(target=read_camera) #start thread to read camera
        t.daemon = True #set thread to daemon so it closes when main thread closes
        t.start()

        brailley.movel([0.172, y_offset, z_depth+0.01, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move above first position
        time.sleep(0.5)
        brailley.movel([0.172, y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move to first position
        time.sleep(0.5)

        print("------------Starting data collection------------\r\n")
        print("------------Dataset size: {}------------\r\n".format(dataset_size))

        move_robot() #move robot to collect data

        print("------------Data collection complete------------\r\n")