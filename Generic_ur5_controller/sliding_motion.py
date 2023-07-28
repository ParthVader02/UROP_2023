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

import os, glob
for filename in glob.glob("frame_*"):
    os.remove(filename) 

with (DigitSensor(serialno='D20654', resolution='QVGA', framerate='60') as digit,
            DisplayImage(window_name='DIGIT Demo') as display): #use wrapper to make accessing DIGIT sensor easier

    tstart = time.time()
    print("------------Configuring braille_bot------------\r\n")
    braille_bot = kgr.kg_robot(port=30000,db_host="169.254.252.50")
    print("----------------Hi braille_bot!-----------------\r\n\r\n")

    ref= cv2.imread("reference_frame.jpg")
    r_ref,g_ref,b_ref = cv2.split(ref)
    def dot_detector(): #function to detect dot positions in frame
        while True:
            frame = digit.get_frame()
            cv2.imwrite(r"frame_{}.jpg".format(digit.frame_count), frame)
            #frame = frame[100:230, 75:150]#crop frame to only show braille area
            #cv2.imshow("frame", frame)
            #cv2.waitKey(1)
            
            h = frame.shape[0] #get height of frame
            w = frame.shape[1] #get width of frame

            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #convert to grayscale

            b,g,r = cv2.split(frame)
            #cv2.imshow("r", cv2.absdiff(r_ref, r))
            #cv2.imshow("g", cv2.absdiff(g_ref, g))
            #cv2.imshow("b", cv2.absdiff(b_ref, b))
             #cv2.imshow("rgb", cv2.absdiff(frame, ref))

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
           
            a,b,c = cv2.split(hsv)
            hsv_split = np.concatenate((a,b,c),axis=1)
            #cv2.imshow("Split HSV",hsv_split)
            #cv2.imwrite(r"frame_{}.jpg".format(digit.frame_count), grey)
            #grey = cv2.medianBlur(grey,5)
            #grey = cv2.GaussianBlur(grey, (5,5),0) #apply gaussian blur to reduce noise
            #grey = cv2.equalizeHist(grey)

            #cv2.imshow("grey", grey)
            cv2.waitKey(1)

            #edges = cv2.Canny(grey,,20) #apply canny edge detection -> keep parameters constant, these give accurate results for hard braille on current elastomer
            #cv2.imwrite(r"frame_{}.jpg".format(digit.frame_count), edges)
            #thresh = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,101,2)
            #ret, thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV)
            #cv2.imwrite(r"frame_{}.jpg".format(digit.frame_count), thresh)
            #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(frame, contours, -1, (0,255,0), 3)

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


    if __name__=='__main__':

        t= Thread(target=dot_detector)
        t.daemon = True

        symbols = []
        headers = ["row_1", "row_2", "row_3"]

        with open("row_data.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            f.close()

          
        braille_bot.movel([0.260394, -0.264857, 0.0135985, 2.09924, 2.33714, -0.000203997], 0.5, 0.2) 
        time.sleep(1)    

        t.start()

       #print(braille_bot.getl()) 
        #braille_bot.movel([-0.178967, -0.468894, -0.0027, -2.30321, -2.1358, -0.00534144], acc=0.1, vel=0.1, wait=True)
        #time.sleep(2)

        time.sleep(0.5)
        braille_bot.movel([0.143181, -0.265351, 0.0135279, 2.16869, 2.27278, -0.00021757], 5, 0.2)
        
        data = pd.read_csv("row_data.csv")
        fig = data.plot().get_figure()
        fig.savefig("data_e.png")
        plt.show()