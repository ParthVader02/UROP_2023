import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
import PIL
def to_tensor_and_normalize(imagepil): #Done with testing
    #This straight up just transforms 0-255 to 0-1
    ChosenTransforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return ChosenTransforms(imagepil)

class SimData(Dataset):
    def __init__(self, setname):
        '''The sim input images and output images are 128 x 128 x 3'''
        self.setname = setname
        assert setname in ['train','val','test']
        
        #Define where to load in the dataset
        overall_dataset_dir = os.path.join(os.path.join(os.getcwd(),'load_dataset'), 'data')
        #input images
        self.selected_dataset_dir = os.path.join(overall_dataset_dir,setname)

        #E.g. self.all_filenames = ['im1.jpg',..,'imN.jpg'] when setname=='train'
        #Loads in the input images from the training folder
        self.all_filenames = os.listdir(self.selected_dataset_dir)

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
        imagepil = imagepil.resize((128,128))

        #convert image to Tensor/normalize
        image = to_tensor_and_normalize(imagepil)
        sample = {'data':image, #preprocessed image, for input into NN
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


model = ConvAutoencoder(layer_disp = True)
model.load_state_dict(torch.load('/home/parth/UROP_2023/Generic_ur5_controller/deblur.pth', map_location=torch.device('cpu')))
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

model.eval()
#Basically load the created class in
test_data = SimData("test")
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, num_workers=0, shuffle = True)

with torch.no_grad():
    dataiter = iter(test_loader)
    #images, labels = next(dataiter)
    output = next(dataiter)
    images = output['data'].to(torch.device('cpu')) #these are the inputs
    pred = model(images) #output of the network
    images = images.detach().cpu().numpy() #convert the images back to numpy arrays

    #convert back to appropriate numpy array and shit
    print(images.shape)
    print(pred.shape)
    predictions = pred.detach().cpu().numpy()

pred = np.transpose(predictions, (1, 2, 0))

cv2.imshow('input',images)
cv2.imshow('preds', pred[1])
cv2.waitKey(0)