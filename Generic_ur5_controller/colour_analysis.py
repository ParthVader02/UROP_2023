import cv2
import numpy as np
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

with (DigitSensor(serialno='D20652', resolution='QVGA', framerate='30') as digit,
            DisplayImage(window_name='DIGIT Demo') as display): #use wrapper to make accessing DIGIT sensor easier


    def dot_detector(frame): #function to detect dot positions in frame
            
        braille_matrix = np.zeros((3,2)) #create empty braille matrix
        
        h = frame.shape[0] #get height of frame
        w = frame.shape[1] #get width of frame
        r,g, b = cv2.split(frame)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        cv2.imshow("hsv", hsv)
        h, s, v = cv2.split(hsv)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        pixel_colors = frame.reshape((np.shape(frame)[0]*np.shape(frame)[1], 3))
        norm = colors.Normalize(vmin=-1.,vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.show()


        #ref = cv2.imread("reference_frame.jpg")
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #convert to grayscale
        #grey_ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY) #convert to grayscale
        #grey = cv2.subtract(grey, grey_ref)
        #grey = cv2.medianBlur(grey,5)
        #grey = cv2.GaussianBlur(grey, (3,3),0) #apply gaussian blur to reduce noise
        #grey = cv2.equalizeHist(grey)

        #edges = cv2.Canny(grey,15,25) #apply canny edge detection -> keep parameters constant, these give accurate results for hard braille on current elastomer
    
        #cv2.imshow("edges", edges)
        #row_1 = edges[0:int(h/3), 0:int(w)]
        #row_2 = edges[int(h/3):int(2*h/3), 0:int(w)]
        #row_3 = edges[int(2*h/3):int(h), 0:int(w)]
        #th3 = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,2)

        #cv2.imshow("thresh", th3)


    if __name__=='__main__':
        
        print("------------Configuring braille_bot------------\r\n")
        #braille_bot = kgr.kg_robot(port=30000,db_host="169.254.252.50")
        print("----------------Hi braille_bot!-----------------\r\n\r\n")

        symbols = []
        headers = ["row_1", "row_2", "row_3"]

        with open("row_data.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            f.close()

       #print(braille_bot.getl()) 
        #braille_bot.movel([-0.178967, -0.468894, -0.0027, -2.30321, -2.1358, -0.00534144], acc=0.1, vel=0.1, wait=True)
        #time.sleep(2)
            
        while True: #len(symbols)<20:
            frame = digit.get_frame()
            frame = frame[100:230, 75:150]#crop frame to only show braille area
            dot_detector(frame)