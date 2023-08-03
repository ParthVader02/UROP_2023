import cv2
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
import time
import random
import os
from threading import Thread
import re

count = 501 #initialise data_count
prev_y = 0
prev_z = 0
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
        base_path = os.path.join(dir_path,"im{}.jpg".format(count)) #create path to save frame
        cv2.imwrite(base_path, frame) 
        
    def move_robot(): #fixed movements for each data collection step
        global count #use global count variable (need to tell function this or it doesnt work) 
        global prev_y, prev_z
        if count%20 ==1: 
            capture_frame("raw_data") #capture frame
        else:
            rand_y = random.uniform(-0.003, 0.003) #generate random y translation
            rand_z = random.uniform(-0.0003, 0) #generate random z translation
            brailley.translatel_rel([0,0,-prev_z,0,0,0], acc=0.5, vel=0.2, wait=True) #remove random z
            time.sleep(0.5)
            brailley.translatel_rel([0, 0, +0.01, 0, 0, 0], 0.5, 0.2) #move up to avoid dragging on surface
            time.sleep(0.5)
            brailley.translatel_rel([0,-prev_y,0,0,0,0], acc=0.5, vel=0.2, wait=True) #remove random y
            time.sleep(0.5)
            brailley.translatel_rel([-0.0058, 0, 0, 0, 0, 0], 0.5, 0.2) #move to next position
            time.sleep(0.5)

            brailley.translatel_rel([0,rand_y,0,0,0,0], acc=0.5, vel=0.2, wait=True) #move to random y position around cell
            time.sleep(0.5) 
            brailley.translatel_rel([0, 0, -0.01, 0, 0, 0], 0.5, 0.2)  #move down to cell
            time.sleep(0.5)
            brailley.translatel_rel([0,0,rand_z,0,0,0], acc=0.5, vel=0.2, wait=True) #move to random z position 
            time.sleep(0.5)

            capture_frame("raw_data") #capture frame
            prev_y = rand_y #store current y to use in next loop
            prev_z = rand_z #store current z to use in next loop
        count += 1 #increment counts

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

    def missing_images(folder_dir, total): #function to check for missing images
        ref_list = list(range(1,total+1)) #create list of reference numbers
        num_list = []
        for image in os.scandir(folder_dir): #loop through images in folder
            path = image.path
            if (path.endswith('.jpg')):
                num = (re.findall(r'\d+', path))[0] #get number from image name
                num_list.append(int(num))
        missing = list(set(ref_list) - set(num_list))  #find missing images
        return missing

    if __name__=='__main__':
        t= Thread(target=read_camera) #start thread to read camera
        t.daemon = True #set thread to daemon so it closes when main thread closes
        t.start()

        brailley.movel([0.290128, -0.271902, 0.02, 2.09818, 2.33554, -0.00188674], 0.5, 0.2) #move above first position
        time.sleep(0.5)
        brailley.movel([0.290128, -0.271902, 0.0172491, 2.09818, 2.33554, -0.00188674], 0.5, 0.2) #move to first position
        time.sleep(0.5)

        print("------------Starting data collection------------\r\n")
        dataset_size = 1000 #set total number of data points to collect

        while count <= dataset_size: 
            print("Data point {} of {} collected".format(count, dataset_size)) #print progress
            if count%21 == 0: #every 20 data points, scroll (21 used as count starts at 1)
                scroll_button() #scroll
                move_robot() #call move robot capture frame after scrolling
            else:
                move_robot() #move robot to next position and capture frame 

        missing = missing_images('raw_data', dataset_size) #check for missing images
        if len(missing) > 0: #if there are missing images
            for i in missing:
                count = i #set count to missing image number
                move_robot() #move robot capture frame
print("------------Data collection complete------------\r\n")