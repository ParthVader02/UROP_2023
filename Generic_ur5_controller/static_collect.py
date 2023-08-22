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

z_depth = 0.015#set z depth of sensor 
y_offset = -0.27 #set y offset of sensor
positions = []
offsets = []
offset_counter= 0

with open("positions.csv", "r") as f:
    reader = csv.reader(f)
    row = next(reader)  # gets the first line
    positions = row
print(len(positions))
#print(positions)
with open("Offsets.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'NaN' and row[0] != '0':
            x = float(row[0])
            y = float(row[1])
            offsets.append([x,y])
        else:
            offsets.append(row)
    #print(offsets)

for path in os.listdir('blurry'):
    # check if current path is a file
    if os.path.isfile(os.path.join('blurry', path)):
        dataset_size += 1
print(dataset_size)

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
        global offsets, offset_counter
        global positions

        for pos in positions:
            if offsets[offset_counter][0] == 'NaN' or offsets[offset_counter][0] == '0':
                offset_counter +=1
                static_count +=1
                print("Frame {} rejected".format(static_count)) 
            else:
                off_x = float(offsets[offset_counter][0])
                off_y = float(offsets[offset_counter][1])
                print(pos)
                print(off_x)
                brailley.movel([float(pos)+(off_x*0.001),y_offset+(off_y*0.001), z_depth+0.01, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move to next position
                time.sleep(0.5)

                brailley.translatel_rel([0, 0, -0.01, 0, 0, 0], 0.5, 0.2)  #move down to cell
                time.sleep(0.5)

                capture_frame("sharp")
                print("Frame {} captured".format(static_count))

                offset_counter+=1

            static_count += 1 #increment counts

        scroll_button() #scroll to next row

    def scroll_button():
        brailley.movel([0.305801, -0.261322, 0.0186874, 2.21758, 2.22249, -0.00198903], 0.5, 0.2) #move to scroll position
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
        
        while static_count <=dataset_size:
            move_robot() #move robot to collect data

        print("------------Data collection complete------------\r\n")