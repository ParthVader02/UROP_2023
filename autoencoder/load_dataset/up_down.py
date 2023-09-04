import cv2
import numpy as np
import csv
import pandas as pd
import time
import matplotlib.pyplot as plt
import sched
import math
from math import pi
import os
import sys
from threading import Thread

sys.path.insert(0, '/home/parth/UROP_2023/Generic_ur5_controller') #add path to brailley controller
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr

capture_flag = False
image_count = 1
tape = True
name = ""

if tape == False:
    name = "no_tape"
else:
    name = "tape"

with (DigitSensor(serialno='D20654', resolution='QVGA', framerate='30') as digit,
            DisplayImage(window_name='DIGIT Demo') as display): #use wrapper to make accessing DIGIT sensor easier

    tstart = time.time()
    print("------------Configuring brailley------------\r\n")
    brailley = kgr.kg_robot(port=30000,db_host="169.254.252.50")
    print("----------------Hi brailley!-----------------\r\n\r\n")

    def read_camera(): #function to read camera
        global frame, capture_flag, image_count, name
        while True:
            frame = digit.get_frame() #get frame from camera
            if capture_flag == True:
                capture_frame("up_down_images/{}".format(name)) #save frame
                capture_flag = False
                image_count += 1 #increment counts
    
    def capture_frame(dir_path): #function to capture frame and save it
        global frame, image_count
        os.makedirs(dir_path, exist_ok=True)  
        base_path = os.path.join(dir_path,"im{}.jpg".format(image_count)) #create path to save frame
        cv2.imwrite(base_path, frame)

    #scheduler used to regularly pass desired positions to servoj 
    scheduler = sched.scheduler(time.time, time.sleep)
    def schedule_it(dt, duration, callable, *args):
        for i in range(int(duration/dt)):
            scheduler.enter(i*dt, 1, callable, args)
    
    #calculate starting position of motion
    def starting_pos(centrepose, xamp, xphase, yamp, yphase, depth, anglex, angley):
        npose = centrepose
        npose = np.add(npose, [0, 0, 0, anglex, angley, 0])
        npose = np.add(npose, [xamp*math.sin(xphase), 0, 0, 0, 0, 0])
        npose = np.add(npose, [0, yamp*math.sin(yphase), 0, 0, 0, 0])
        brailley.movel(npose)
        
    # main function: moves to desired position at any moment in time
    def parameter_move(t0, centrepose, xamp, xfreq, xphase, yamp, yfreq, yphase, zamp, zfreq, depth, anglex, angley, decayrate):
        global capture_flag
        #start with z height
        npose = centrepose
        #add angles
        npose = np.add(npose, [0, 0, 0, anglex, angley, 0])
        t = time.time() - t0
        #xamp = xamp*math.exp(-decayrate*t)
        #yamp = yamp*math.exp(-decayrate*t)
        #zamp = zamp*math.exp(-decayrate*t)
        xamp = max(0,(1-decayrate*t)*xamp) 
        yamp = max(0,(1-decayrate*t)*yamp)
        zamp = max(0,(1-decayrate*t)*zamp)
        #x vibrations
        npose = np.add(npose, [xamp*math.sin(xfreq*t+xphase), 0, 0, 0, 0, 0])
        #y vibrations
        npose = np.add(npose, [0, yamp*math.sin(yfreq*t+yphase), 0, 0, 0, 0])
        #zvibrations
        npose = np.add(npose, [0, 0, zamp*math.sin(zfreq*t), 0, 0, 0])
        #pass to UR5
        brailley.servoj(npose, vel=50, control_time=0.05)
        capture_flag = True

    if __name__=='__main__':

        t= Thread(target=read_camera)
        t.daemon = True
        t.start()

        #1hz
        brailley.movel([0.292, -0.27, 0.0148, 2.21745, 2.22263, -0.00101733], 0.5, 0.2) #move above tub
        centrepose=[0.292, -0.27, 0.0148, 2.21745, 2.22263, -0.00101733]

        #5hz
        #braille_bot.movel([0.259405, -0.26263, 0.019, 2.09924, 2.33716, -0.000108163], 0.5, 0.02) #move above tub
        #centrepose=[0.259405, -0.26263, 0.019, 2.09924, 2.33716, -0.000108163]

        #move to starting position
        starting_pos(centrepose, 0, 0, 0, 0, 0, 0, 0)
        time.sleep(0.5)

        t0 = time.time()
        #initialise scheduler
        #schedule_it(0.05, 10, parameter_move, t0, centrepose,0, 0, 0, 0, 0, 0, 0.05, 2*pi*5, 0, 0, 0, 0)
        schedule_it(0.05, 5, parameter_move, t0, centrepose,0, 0, 0, 0, 0, 0, 0.005, 2*pi*1, 0, 0, 0, 0)

        #run scheduler calling servoj
        scheduler.run()
