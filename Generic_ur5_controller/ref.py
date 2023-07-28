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

            frame = frame[100:230, 75:150]#crop frame to only show braille area
            
            h = frame.shape[0] #get height of frame
            w = frame.shape[1] #get width of frame

            #ref = cv2.imread("reference_frame.jpg")
            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #convert to grayscale
            #grey_ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY) #convert to grayscale
            #grey = cv2.subtract(grey, grey_ref)
            #grey = cv2.medianBlur(grey,5)
            #grey = cv2.GaussianBlur(grey, (3,3),0) #apply gaussian blur to reduce noise
            #grey = cv2.equalizeHist(grey)
            #grey = cv2.Laplacian(grey,cv2.CV_64F, ksize=15)
            #grey = cv2.Sobel(grey,cv2.CV_64F,1,0,ksize=5)
            #grey = cv2.Sobel(grey,cv2.CV_64F,0,1,ksize=5)
            edges = cv2.Canny(grey,15,25) #apply canny edge detection -> keep parameters constant, these give accurate results for hard braille on current elastomer
        
            #cv2.imshow("grey", grey)
            #cv2.waitKey(1)

            row_1 = edges[0:int(h/3), 0:int(w)]
            row_2 = edges[int(h/3):int(2*h/3), 0:int(w)]
            row_3 = edges[int(2*h/3):int(h), 0:int(w)]
            #th3 = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,2)

            #cv2.imshow("thresh", th3)


            #row_1 = grey[0:int(h/3), 0:int(w)]
            #row_2 = grey[int(h/3):int(2*h/3), 0:int(w)]
            #row_3 = grey[int(2*h/3):int(h), 0:int(w)]

            #cv2.imshow("grey", row_3)
            #cv2.waitKey(1)

            #row_1 = th3[0:int(h/3), 0:int(w)]
            #row_2 = th3[int(h/3):int(2*h/3), 0:int(w)]
            #row_3 = th3[int(2*h/3):int(h), 0:int(w)]

            avg_1 = np.mean(row_1)/np.sum(grey)
            avg_2 = np.mean(row_2)/np.sum(grey)
            avg_3 = np.mean(row_3)/np.sum(grey)
            
            row = [avg_1, avg_2, avg_3]
            with open("row_data_ref.csv", "a") as f:
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

        with open("row_data_ref.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            f.close()

        t.start()

       #print(braille_bot.getl()) 
        #braille_bot.movel([-0.178967, -0.468894, -0.0027, -2.30321, -2.1358, -0.00534144], acc=0.1, vel=0.1, wait=True)
        #time.sleep(2)

        braille_bot.movel([0.259405, -0.26263, 0.022, 2.09924, 2.33716, -0.000108163], 0.5, 0.02) #move above tub
        centrepose=[0.259405, -0.26263, 0.022, 2.09924, 2.33716, -0.000108163]

        #move to starting position
        starting_pos(centrepose, 0, 0, 0, 0, 0, 0, 0)
        time.sleep(0.5)

        t0 = time.time()
        #initialise scheduler
        schedule_it(0.05, 10, parameter_move, t0, centrepose,0, 0, 0, 0, 0, 0, 0.009, 2*pi*0.5, 0, 0, 0, 0)

        #run scheduler calling servoj
        scheduler.run()
        #t.join()
        data = pd.read_csv("row_data_ref.csv")
        fig = data.plot().get_figure()
        fig.savefig("ref.png")
        plt.show()