import cv2
import numpy as np
from digit_demo import DigitSensor, DisplayImage
from statistics import mode
import kg_robot as kgr
import time

with (DigitSensor(serialno='D20654', resolution='QVGA', framerate='30') as digit,
            DisplayImage(window_name='DIGIT Demo') as display): #use wrapper to make accessing DIGIT sensor easier


    def dot_detector(frame): #function to detect dot positions in frame
            
            braille_matrix = np.zeros((3,2)) #create empty braille matrix
            
            h = frame.shape[0] #get height of frame
            w = frame.shape[1] #get width of frame

            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #convert to grayscale

            grey = cv2.GaussianBlur(grey, (5,5),0) #apply gaussian blur to reduce noise
            cv2.imshow("blur", grey)
            edges = cv2.Canny(grey,13,25) #apply canny edge detection
            cv2.imshow("canny edge", edges)

            circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, minDist=40,
            param1=15, param2=9,
            minRadius=14, maxRadius=17) #apply hough transform to detect circles in frame

            if circles is not None:  #if circles are detected
                circles = np.uint16(np.around(circles)) #convert to integer array

                for i in circles[0, :]:
                    centre = (i[0], i[1]) #get centre of circle
                    if centre[0] < w/2 and centre[1] < h/3: #if centre is in rectangle 1
                        braille_matrix[0,0] = 1
                    elif centre[0] < w/2 and h/3 < centre[1] < 2*h/3: #if centre is in rectangle 2
                        braille_matrix[1,0] = 1
                    elif centre[0] < w/2 and 2*h/3 < centre[1] < h :#if centre is in rectangle 3
                        braille_matrix[2,0] = 1
                    elif centre[0] > w/2 and centre[1] < h/3: #if centre is in rectangle 4
                        braille_matrix[0,1] = 1
                    elif centre[0] > w/2 and h/3 < centre[1] < 2*h/3: #if centre is in rectangle 5
                        braille_matrix[1,1] = 1
                    elif centre[0] > w/2 and 2*h/3 < centre[1] < h: #if centre is in rectangle 6
                        braille_matrix[2,1] = 1
                    cv2.circle(frame, centre, 1, (0, 100, 100), 3) #draw centres of circle
                    #cv2.circle(frame, centre, i[2], (255, 0, 255), 3) #draw circles
            if circles is None: 
                braille_matrix = np.zeros((3,2)) #if no circles are detected, return empty matrix
            return braille_matrix

    def braille_to_english(braille):
        braille_string = my_string = ','.join(str(x) for x in braille)
        braille_dict = ({"a": "[1. 0.],[0. 0.],[0. 0.]", "b":"[1. 0.],[1. 0.],[0. 0.]", "c":"[1. 1.],[0. 0.],[0. 0.]",
                         "d":"[1. 1.],[0. 1.],[0. 0.]", "e":"[1. 0.],[0. 1.],[0. 0.]", "f":"[1. 1.],[1. 0.],[0. 0.]", 
                         "g":"[1. 1.],[1. 1.],[0. 0.]", "h":"[1. 0.],[1. 1.],[0. 0.]", "i":"[0. 1.],[1. 0.],[0. 0.]", 
                         "j":"[0. 1.],[1. 1.],[0. 0.]", "k":"[1. 0.],[0. 0.],[1. 0.]", "l":"[1. 0.],[1. 0.],[1. 0.]", 
                         "m":"[1. 1.],[0. 0.],[1. 0.]", "n":"[1. 1.],[0. 1.],[1. 0.]", "o":"[1. 0.],[0. 1.],[1. 0.]", 
                         "p":"[1. 1.],[1. 0.],[1. 0.]","q":"[1. 1.],[1. 1.],[1. 0.]", "r":"[1. 0.],[1. 1.],[1. 0.]", 
                         "s":"[0. 1.],[1. 0.],[1. 0.]", "t":"[0. 1.],[1. 1.],[1. 0.]", "u":"[1. 0.],[0. 0.],[1. 1.]",
                         "v":"[1. 0.],[1. 0.],[1. 1.]", "w":"[0. 1.],[1. 1.],[0. 1.]", "x":"[1. 1.],[0. 0.],[1. 1.]", 
                         "y":"[1. 1.],[0. 1.],[1. 1.]", "z":"[1. 0.],[0. 1.],[1. 1.]", " ":"[0. 0.],[0. 0.],[0. 0.]"}) 

        if braille_string in braille_dict.values():
            return list(braille_dict.keys())[list(braille_dict.values()).index(braille_string)]

    if __name__=='__main__':
        
        print("------------Configuring brailley------------\r\n")
        brailley = kgr.kg_robot(port=30000,db_host="169.254.252.50")
        print("----------------Hi brailley!-----------------\r\n\r\n")

        symbols = []
        print(brailley.getl()) 
        #brailley.movel([0.260394, -0.264857, 0.0135985, 2.09924, 2.33714, -0.000203997], 0.5, 0.2) 
        #time.sleep(2)

        while True: #len(symbols)<20:
            #brailley.translatel_rel([0,0,-0.002,0,0,0], acc=0.05, vel=0.1, wait=True)
            #brailley.servoj([0.259341, -0.262522, 0.0816393+0.01, -2.0993, -2.33719, 6.73604e-05], vel=1)
            #brailley.servoj([0.259341, -0.262522, 0.0816393-0.01, -2.0993, -2.33719, 6.73604e-05], vel=1)
           
            letters = [] #create empty list to store letters
            while digit.frame_count <=59:
                frame = digit.get_frame()
                #frame = frame[100:230, 75:150]#crop frame to only show braille area
                braille_matrix = dot_detector(frame)
                english = braille_to_english(braille_matrix)
                if english != None:
                    letters.append(english) #add letter to list
                display.show_image([frame],window_scale=1)
                # '27' is the escape key
                if cv2.waitKey(1)==27:
                    break
            if digit.frame_count == 60: 
                if len(letters) > 0:
                    symbols.append(mode(letters))
                digit.frame_count = 0
            #brailley.translatel_rel([0,0,+0.002,0,0,0], acc=0.05, vel=0.1, wait=True)
            #brailley.translatel_rel([0,+0.0063,0,0,0,0], acc=0.05, vel=0.1, wait=True)
                
            print(symbols)
            

        if len(symbols) == 20:
            #brailley.movel([0.3649,-0.0245,-0.4039,0.58,-3.09,0], acc=0.1, vel=0.1, wait=True)
            time.sleep(2)
            braille_bot.translatel_rel([0,0,+0.02,0,0,0], acc=0.1, vel=0.1, wait=True)
            time.sleep(2)
            #brailley.translatel_rel([0,0,0.001,0,0,0], acc=0.1, vel=0.1, wait=True)
            braille_bot.close()
        
