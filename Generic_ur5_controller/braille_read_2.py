import cv2
import numpy as np
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
import csv
import pandas as pd
import time
import matplotlib.pyplot as plt
import sched
import math
from math import pi
from threading import Thread

with (DigitSensor(serialno='D20654', resolution='QVGA', framerate='30') as digit,
            DisplayImage(window_name='DIGIT Demo') as display): #use wrapper to make accessing DIGIT sensor easier

    tstart = time.time()
    print("------------Configuring braille_bot------------\r\n")
    braille_bot = kgr.kg_robot(port=30000,db_host="169.254.252.50")
    print("----------------Hi braille_bot!-----------------\r\n\r\n")

    #scheduler used to regularly pass desired positions to servoj 
    scheduler = sched.scheduler(time.time, time.sleep)
    def schedule_it(dt, duration, callable, *args):
        for i in range(int(duration/dt)):
            scheduler.enter(i*dt, 1, callable, args)
    
    def dot_detector(): #function to detect dot positions in frame
        while True:
            frame = digit.get_frame()
            #cv2.imwrite("images/frame_{}.png".format(digit.frame_count), frame)
            #frame = frame[100:230, 75:150]#crop frame to only show braille area
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            print(frame.shape)
            h = frame.shape[0] #get height of frame
            w = frame.shape[1] #get width of frame

            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #convert to grayscale

            #grey = cv2.medianBlur(grey,5)
            #grey = cv2.GaussianBlur(grey, (3,3),0) #apply gaussian blur to reduce noise
            #grey = cv2.equalizeHist(grey)

            #cv2.imshow("grey", grey)
            #cv2.waitKey(1)

            #edges = cv2.Canny(grey,15,25) #apply canny edge detection -> keep parameters constant, these give accurate results for hard braille on current elastomer
            circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 2, minDist=40,
            param1=20, param2=9,
            minRadius=7, maxRadius=15) #apply hough transform to detect circles in frame
            if circles is not None:  #if circles are detected
                circles = np.uint16(np.around(circles)) #convert to integer array
                for i in circles[0, :]:
                    centre = (i[0], i[1]) #get centre of circle
                    cv2.circle(frame, centre, 1, (0, 100, 100), 3) #draw centres of circles
            cv2.imwrite(r"frame_{}.jpg".format(digit.frame_count), frame)
            #row_1 = edges[0:int(h/3), 0:int(w)]
            #row_2 = edges[int(h/3):int(2*h/3), 0:int(w)]
            #row_3 = edges[int(2*h/3):int(h), 0:int(w)]
            #th3 = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,2)

            #cv2.imshow("thresh", th3)
            #cv2.waitKey(1)

            row_1 = grey[0:int(h/3), 0:int(w)]
            row_2 = grey[int(h/3):int(2*h/3), 0:int(w)]
            row_3 = grey[int(2*h/3):int(h), 0:int(w)]

            #cv2.imshow("grey", row_1)
            #cv2.waitKey(1)

            #row_1 = th3[0:int(h/3), 0:int(w)]
            #row_2 = th3[int(h/3):int(2*h/3), 0:int(w)]
            #row_3 = th3[int(2*h/3):int(h), 0:int(w)]

            avg_1 = np.mean(row_1)/np.sum(grey)
            avg_2 = np.mean(row_2)/np.sum(grey)
            avg_3 = np.mean(row_3)/np.sum(grey)
            
            
            row = [avg_1, avg_2, avg_3]
            with open("row_data.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.close()       

    #calculate starting position of motion
    def starting_pos(centrepose, xamp, xphase, yamp, yphase, depth, anglex, angley):
        npose = centrepose
        npose = np.add(npose, [0, 0, 0, anglex, angley, 0])
        npose = np.add(npose, [xamp*math.sin(xphase), 0, 0, 0, 0, 0])
        npose = np.add(npose, [0, yamp*math.sin(yphase), 0, 0, 0, 0])
        braille_bot.movel(npose)
        
    # main function: moves to desired position at any moment in time
    def parameter_move(t0, centrepose, xamp, xfreq, xphase, yamp, yfreq, yphase, zamp, zfreq, depth, anglex, angley, decayrate):
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
        braille_bot.servoj(npose, vel=50, control_time=0.05)

    if __name__=='__main__':

        t= Thread(target=dot_detector)
        t.daemon = True

        symbols = []
        headers = ["row_1", "row_2", "row_3"]

        with open("row_data.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            f.close()

        t.start()

       #print(braille_bot.getl()) 
        #braille_bot.movel([-0.178967, -0.468894, -0.0027, -2.30321, -2.1358, -0.00534144], acc=0.1, vel=0.1, wait=True)
        #time.sleep(2)

        #1hz
        braille_bot.movel([0.259405, -0.26263, 0.019, 2.09924, 2.33716, -0.000108163], 0.5, 0.2) #move above tub
        centrepose=[0.259405, -0.26263, 0.019, 2.09924, 2.33716, -0.000108163]

        #5hz
        #braille_bot.movel([0.259405, -0.26263, 0.019, 2.09924, 2.33716, -0.000108163], 0.5, 0.02) #move above tub
        #centrepose=[0.259405, -0.26263, 0.019, 2.09924, 2.33716, -0.000108163]

        #move to starting position
        starting_pos(centrepose, 0, 0, 0, 0, 0, 0, 0)
        time.sleep(0.5)

        t0 = time.time()
        #initialise scheduler
        #schedule_it(0.05, 10, parameter_move, t0, centrepose,0, 0, 0, 0, 0, 0, 0.05, 2*pi*5, 0, 0, 0, 0)
        schedule_it(0.05, 10, parameter_move, t0, centrepose,0, 0, 0, 0, 0, 0, 0.01, 2*pi*1, 0, 0, 0, 0)

        #run scheduler calling servoj
        scheduler.run()

        data = pd.read_csv("row_data.csv")
        fig = data.plot().get_figure()
        fig.savefig("data_e.png")
        plt.show()