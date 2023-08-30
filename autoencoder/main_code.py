import os
import PIL
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import cv2

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Function for showing images
def imshow(img):
    #img = img / 2 + 0.5  
    #print(img)
    #plt.imshow can take in images with 0-1 (floats) or 0-255 (int)
    plt.imshow(np.transpose(img, (1, 2, 0))) 
    #print(img)
    #print(max(img[0][0]))
    print(np.mean(img[0][2]))
    print(np.std(img[0][2]))
    #print(max(img[0][1]))
    #print(max(img[0][2]))
#It's best to keep the data processing code separate from the class loader
def to_tensor_and_normalize(imagepil): #Done with testing
    """Convert image to torch Tensor and normalize using the ImageNet training
    set mean and stdev taken from
    https://pytorch.org/docs/stable/torchvision/models.html.
    Why the ImageNet mean and stdev instead of the PASCAL VOC mean and stdev?
    Because we are using a model pretrained on ImageNet."""
    #Think the reason for introducing normalisation is because of the imagenet weights
    #ChosenTransforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #            torchvision.transforms.Normalize(mean=[0.6236, 0.5118, 0.4264],std=[0.3545, 0.2692, 0.3376]),])


    #This straight up just transforms 0-255 to 0-1
    ChosenTransforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])#, torchvision.transforms.Normalize(mean=[0.3, 0.3, 0.3],std=[0.05, 0.05, 0.05])])
    #ChosenTransforms = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])#[torchvision.transforms.ToTensor()])
    """ChosenTransforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      lambda x: x*1.666666666666666666667,
                                                      ])"""
    return ChosenTransforms(imagepil)
class SimData(Dataset):
    def __init__(self, setname):
        '''The sim input images and output images are 128 x 128 x 3'''
        self.setname = setname
        assert setname in ['train','val','test']
        
        #Define where to load in the dataset
        overall_dataset_dir = os.path.join(os.path.join('/home/parth/UROP_2023/autoencoder','load_dataset'), 'data')
        #input images
        self.selected_dataset_dir = os.path.join(overall_dataset_dir,setname)
        
        #output images
        self.selected_dataset_output_dir = os.path.join(overall_dataset_dir,setname+"_outputs")

        count = 0
        # Iterate directory
        for path in os.listdir(self.selected_dataset_dir): #get number of images in folder
            # check if current path is a file
            if os.path.isfile(os.path.join(self.selected_dataset_dir, path)):
                count += 1 # increment count
        
        #E.g. self.all_filenames = ['im1.jpg',..,'imN.jpg'] when setname=='train'
        #Loads in the input images from the training folder
        self.all_filenames = [] #initalise list of filenames
        self.all_filenames_output = [] 
        for i in range(1,count+1):  #iterate through all the images by number rather than os.listdir which does it randomly
            self.all_filenames.append("im"+str(i)+".jpg")
            self.all_filenames_output.append("im"+str(i)+".jpg")
        
    def __len__(self):
        """Return the total number of examples in this split, e.g. if
        self.setname=='train' then return the total number of examples
        in the training set"""
        return len(self.all_filenames)
    
    def __getitem__(self,idx):
        """Return the example at index [idx]. The example is a dict with keys
        'data' (value: Tensor for an RGB image) and labels are also images"""
        #For the inputs
        selected_filename = self.all_filenames[idx]
        #print(selected_filename)
        #test= self.all_filenames_output[idx]
        #print(test)
        imagepil = PIL.Image.open(os.path.join(self.selected_dataset_dir,selected_filename)).convert('RGB')
        
        #For the outputs
        selected_filename_output = self.all_filenames_output[idx]
        imagepil_output = PIL.Image.open(os.path.join(self.selected_dataset_output_dir,
                                                      selected_filename_output)).convert('RGB')
        
        #convert image to Tensor/normalize
        image = to_tensor_and_normalize(imagepil)
        image_output = to_tensor_and_normalize(imagepil_output)
        
        
        sample = {'data':image, #preprocessed image, for input into NN
                  'label':image_output,
                  'img_idx':idx}
        
        return sample

class ConvAutoencoder(nn.Module):
    def __init__(self, layer_disp = False):
        super(ConvAutoencoder, self).__init__()
        self.layer_disp = layer_disp
        #Encoder
        #nn.Conv2(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=128,
                               kernel_size=2, 
                               stride = 1,
                               padding = 'same')
        self.conv2 = nn.Conv2d(in_channels=128, 
                               out_channels=256,
                               kernel_size=2, 
                               stride = 2,
                               padding = 0)
        
        self.conv3 = nn.Conv2d(in_channels=256, 
                               out_channels=256,
                               kernel_size=4, 
                               stride = 4,
                               padding = 0)
        
        
        
        self.pool = nn.MaxPool2d(2,2)
        
       
        #Decoder
        #nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.t_conv1 = nn.ConvTranspose2d(in_channels=256, 
                               out_channels=128,
                               kernel_size=2, 
                               stride = 2,
                               padding=0)
        self.t_conv2= nn.ConvTranspose2d(in_channels=128, 
                               out_channels=128,
                               kernel_size=2, 
                               stride = 2,
                               padding=0)
        self.t_conv3= nn.ConvTranspose2d(in_channels=128, 
                               out_channels=64,
                               kernel_size=2, 
                               stride = 2,
                               padding=0)
        
        self.t_conv4= nn.ConvTranspose2d(in_channels=64, 
                               out_channels=64,
                               kernel_size=2, 
                               stride = 2,
                               padding=0)
        self.t_conv5= nn.ConvTranspose2d(in_channels=64, 
                               out_channels=32,
                               kernel_size=2, 
                               stride = 2,
                               padding=0)
        '''self.t_conv6= nn.ConvTranspose2d(in_channels=32, 
                               out_channels=32,
                               kernel_size=2, 
                               stride = 2,
                               padding=0)'''
       
        self.t_convout = nn.ConvTranspose2d(in_channels=32, 
                               out_channels=3,
                               kernel_size=2, 
                               stride = 2,
                               padding=0)
        #self.upsample = nn.functional.interpolate(scale_factor = 2)


    def forward(self, x):
        en_layer1 = F.relu(self.conv1(x))   
        en_layer1 = self.pool(en_layer1)
        en_layer2 = F.relu(self.conv2(en_layer1))
        en_layer2 = self.pool(en_layer2)
        en_layer3 = F.relu(self.conv3(en_layer2))
        en_layer3 = self.pool(en_layer3)

        #de_layer0 = F.relu(self.t_conv0(en_layer4)) 
        de_layer1 = F.relu(self.t_conv1(en_layer3))   
        #de_layer1 = nn.functional.interpolate(de_layer1,scale_factor = 2)
        de_layer2= F.relu(self.t_conv2(de_layer1)) 
        de_layer3= F.relu(self.t_conv3(de_layer2))
        de_layer4= F.relu(self.t_conv4(de_layer3))
        de_layer5= F.relu(self.t_conv5(de_layer4))
        #de_layer6= F.relu(self.t_conv6(de_layer5))
        
        #de_layerout = torch.sigmoid(self.t_conv3(de_layer2))
        de_layerout = F.relu(self.t_convout(de_layer5))
        #de_layerout = self.t_conv5(de_layer4)
        
        if self.layer_disp:
            print("input", x.shape)
            print("en_layer1",en_layer1.shape)
            #print("en_layer1a",en_layer1a.shape)
            print("en_layer2",en_layer2.shape)
            #print("en_layer2a",en_layer2a.shape)
            print("en_layer3",en_layer3.shape)
            #print("en_layer3a",en_layer3a.shape)
            #print("en_layer4",en_layer4.shape)
            #print("en_layer5",en_layer5.shape)
            
            #print("de_layer0",de_layer0.shape)
            print("de_layer1",de_layer1.shape)
            #print("de_layer1a",de_layer1a.shape)
            print("de_layer2",de_layer2.shape)
            print("de_layer3",de_layer3.shape)
            print("de_layer4",de_layer4.shape)
            print("de_layer5",de_layer5.shape)
            #print("de_layer6",de_layer6.shape)
            #print("de_layer2a",de_layer2a.shape)
            #print("de_layer3",de_layerou.shape)
            #print("de_layer3a",de_layer3a.shape)
            #print("de_layer4",de_layer4.shape)
            print("de_layerout",de_layerout.shape)
              
        return de_layerout

#Now to intialise the model 
model = ConvAutoencoder(layer_disp = False).to(device)
model = ConvAutoencoder(layer_disp = True).to(device)

#Defining the loss function between the input and the output
criterion = nn.MSELoss()
#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1*1e-3)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma=0.5)

#This is just for testing and seeing the outputs of each convolutional layer
dummy = model(torch.empty(3, 128, 128).to(device))

#testing code:
##############################################################################################################
#import libraries

import cv2
import sys
import time
import os
from threading import Thread
import re
import csv
from roboflow import Roboflow
import gc
gc.collect()

sys.path.insert(0, '/home/parth/UROP_2023/Generic_ur5_controller') #add path to brailley controller
from digit_demo import DigitSensor, DisplayImage
import kg_robot as kgr
##############################################################################################################
#get target text file properties

with open('/home/parth/UROP_2023/test_properties.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        props = row
    f.close()

print(props)
word_count = int(props[0])
letter_count = int(props[1])
line_count = int(props[2])
##############################################################################################################
#read target text file on braille reader

# total_rows = line_count  #found from target text file
# dynamic_count = 1
# row_counter = 1

# z_depth = 0.0143 #set z depth of sensor, with medical tape need to be lower for clarity
# y_offset = -0.27 #set y offset of sensor

# velocity = 0.2 #initialise velocity
# start = 0
# end = 0
# time_taken = 0
# slide_capture_flag = False #initialise slide capture flag

# with DigitSensor(serialno='D20654', resolution='QVGA', framerate='60') as digit:
#     frame = digit.get_frame()
#     print("------------Configuring brailley------------\r\n")
#     brailley = kgr.kg_robot(port=30000,db_host="169.254.252.50")
#     print("----------------Hi brailley!----------------\r\n\r\n")

#     def read_camera(): #function to read camera
#          global frame
#          global start
#          while True:
#             frame = digit.get_frame() #get frame from camera
#             if slide_capture_flag == True: 
#                 capture_frame("/home/parth/UROP_2023/autoencoder/test_blurry") #capture frame

#     def capture_frame(dir_path): #function to capture frame and save it
#         global static_collect, dynamic_collect, static_count, dynamic_count
#         os.makedirs(dir_path, exist_ok=True)  
#         base_path = os.path.join(dir_path,"im{}.jpg".format(dynamic_count)) #create path to save frame
#         cv2.imwrite(base_path, frame)
#         dynamic_count += 1 #increment counts
    
#     def move_robot(): #fixed movements for each data collection step
#         global dynamic_count, velocity, row_counter
#         global slide_capture_flag
#         global start, time_taken, end
#         start = time.time() #start timer
#         slide_capture_flag = True
#         brailley.movel([0.296, y_offset, z_depth,  2.21745, 2.22263, -0.00201733], 500, velocity) #slide across one row
#         slide_capture_flag = False
#         end = time.time()
#         time.sleep(0.1)
#         t = end - start #calculate time taken to slide across row
#         time_taken += t #add time to total time

#         row_counter += 1 #increment row counter
#         scroll_button() #press scroll button

#     def scroll_button():
#         brailley.movel([0.305801, -0.261322, 0.0186874, 2.21758, 2.22249, -0.00198903], 5, 0.2) #move to scroll position
#         time.sleep(0.1)
#         brailley.translatel_rel([0, 0, -0.006, 0, 0, 0], 0.5, 0.2) #press scroll button
#         time.sleep(0.1)
#         brailley.translatel_rel([0, 0, 0.006, 0, 0, 0], 0.5, 0.2) #move back to scroll position
#         time.sleep(0.1)
#         brailley.movel([0.17,y_offset, z_depth+0.01, 2.21745, 2.22263, -0.00201733], 0.6, 0.4) #move above first position
#         time.sleep(0.1)
#         brailley.movel([0.17,y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.4) #move to first position
#         time.sleep(0.1)

#     if __name__=='__main__':
#         t= Thread(target=read_camera) #start thread to read camera
#         t.daemon = True #set thread to daemon so it closes when main thread closes
#         t.start()
        
#         brailley.movel([0.17, y_offset, z_depth+0.01, 2.21745, 2.22263, -0.00101733], 0.5, 0.2) #move above first position
#         time.sleep(0.5)
#         brailley.movel([0.17, y_offset, z_depth, 2.21745, 2.22263, -0.00201733], 0.5, 0.2) #move to first position
#         time.sleep(0.5)

#         print("------------Starting data collection------------\r\n")
        
#         while row_counter <= total_rows: #get at least the target data set size
#             move_robot() #movements
#             print("Row {} of {} collected".format(row_counter, total_rows)) #print progress
        
#         wpm_speed = word_count/(time_taken/60) #calculate words per minute
#         print("Time taken: {} seconds".format(time_taken)) #print time taken
#         print("Words: {}".format(word_count))
#         print("Speed: {} words per minute".format(wpm_speed))
# print("------------Data collection complete------------\r\n") 
##############################################################################################################
#process images for autoencoder

for image in os.scandir('/home/parth/UROP_2023/autoencoder/test_blurry'): #for each image in the blurry folder
    path = image.path
    num = (re.findall(r'im(\d+)', path))[0] #get image number

    img = cv2.imread(path) #read image
    new_size = (128, 128) # new_size=(width, height)
    img = cv2.resize(img, new_size) 

    os.makedirs('/home/parth/UROP_2023/autoencoder/load_dataset/data/test', exist_ok=True) #save resized images to test folder
    base_path = os.path.join('/home/parth/UROP_2023/autoencoder/load_dataset/data/test',"im{}.jpg".format(num)) 
    cv2.imwrite(base_path, img)

    os.makedirs('/home/parth/UROP_2023/autoencoder/load_dataset/data/test_outputs', exist_ok=True) #need to create test_outputs folder to work with SimData class
    base_path = os.path.join('/home/parth/UROP_2023/autoencoder/load_dataset/data/test_outputs',"im{}.jpg".format(num)) 
    cv2.imwrite(base_path, img)
##############################################################################################################
#load blurry images into autoencoder

test_dataset = SimData("test") #load test dataset as SimData class
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, num_workers=0, shuffle = False) #no shuffle so images go in order

model.load_state_dict(torch.load("/home/parth/UROP_2023/deblur.pth", map_location=torch.device('cpu'))) #load pre-trained autoencdoer model
model.eval() #set model to evaluation mode

#input blurry images into autoencoder
with torch.no_grad():
    dataiter = iter(test_loader)
    output = next(dataiter)
    images = output['data'].to(device) #these are the inputs
    labels = output['label'].to(device)

    #Sample outputs
    predictions = model(images) #output of the network
    images = images.detach().cpu().numpy() #convert the images back to numpy arrays
    labels = labels.detach().cpu().numpy() #convert the images back to numpy arrays

    #convert back to appropriate numpy array 
    print(images.shape)
    print(predictions.shape)
    predictions = predictions.detach().cpu().numpy()

    for idx in np.arange(len(test_dataset)): 
        #save input images
        input = np.transpose(images[idx], (1, 2, 0)) #convert to correct format
        input = np.array(255*input, dtype = 'uint8') 
        input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR) 
        os.makedirs('/home/parth/UROP_2023/autoencoder/inputs', exist_ok=True) 
        base_path = os.path.join('/home/parth/UROP_2023/autoencoder/inputs',"im{}.jpg".format(idx+1)) 
        cv2.imwrite(base_path,input)

        #save output images
        pred = np.transpose(predictions[idx], (1, 2, 0)) #convert to correct format
        pred = np.array(255*pred, dtype = 'uint8')
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        os.makedirs('/home/parth/UROP_2023/autoencoder/predictions', exist_ok=True) 
        base_path = os.path.join('/home/parth/UROP_2023/autoencoder/predictions',"im{}.jpg".format(idx+1))
        cv2.imwrite(base_path,pred) 

##############################################################################################################
#input deblurred images into pre-trained YOLO v8 model

rf = Roboflow(api_key="qnNaV5QQtZevwcJxsA5Y") #create roboflow object
project = rf.workspace().project("digit-braille") 
classifier = project.version(4).model #load pre-trained model
pred_text = "" #initialise string of predicted text

os.makedirs('/home/parth/UROP_2023/autoencoder/braille_detect', exist_ok=True) #make folder to store braille classifier images
for i in range(1, len(test_dataset)+1): #for each image in the predictions folder
    #infer on each image
    pred_dict = classifier.predict("/home/parth/UROP_2023/autoencoder/predictions/im{}.jpg".format(i), confidence=0, overlap=30).json() #save as dictionary data type
    if 'predictions' in pred_dict:
        preds = pred_dict['predictions'] #get predictions from dictionary
        max_conf = 0 #initialise max confidence
        if len(preds) > 0:
            for pred in preds:
                    if pred['confidence'] > max_conf: #if confidence is higher than previous max confidence
                        max_class = pred['class'] #save prediction with highest confidence
            pred_text += max_class #add predicted character with highest confidence to string
        else:
            pred_text += " " #add space if no predictions
    #save bounding box image to braille_detect folder
    classifier.predict("/home/parth/UROP_2023/autoencoder/predictions/im{}.jpg".format(i), confidence=0, overlap=30).save("/home/parth/UROP_2023/autoencoder/braille_detect/detect{}.jpg".format(i))
##############################################################################################################
#Compare predicted text with target text
pred_list = list(pred_text)
for i in range(line_count+1):
    pred_list[40*i:(40*i)+40] = pred_list[40*i:(40*i)+40][::-1] #reverse each line of predicted text, 40 frames per row with 0.2m/s
pred_text = "".join(pred_list) #convert back to string
print(pred_text)