import os
import cv2
import numpy as np
import re

# folder path
noisy_path = 'noisy_data'
total_size = 0
count = 0

# Iterate directory
for path in os.listdir(noisy_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(noisy_path, path)):
        total_size += 1

train_ratio = 0.8 #set ratio of training data to validation data

train_size = int(total_size*train_ratio) #calculate number of training images
val_size = total_size - train_size #calculate number of validation images
print("Total size: {}\r\nTrain size: {}\r\nValidation size: {}".format(total_size, train_size, val_size))

for image in os.scandir(noisy_path):
    path = image.path
    num = (re.findall(r'\d+', path))[0]
    input_image = cv2.imread(path)
    output_image = cv2.imread('raw_data/im{}.jpg'.format(num)) #read in image

    # resize the images for training on existing autoencoder
    new_size = (128, 128) # new_size=(width, height)
    input_image = cv2.resize(input_image, new_size)
    output_image = cv2.resize(output_image, new_size)

    if count <= train_size: #if image is to be in training set
        os.makedirs('train_inputs', exist_ok=True) 
        base_path = os.path.join('train_inputs',"im{}.jpg".format(num)) #training inputs
        cv2.imwrite(base_path, input_image)

        os.makedirs('train_outputs', exist_ok=True) 
        base_path = os.path.join('train_outputs',"im{}.jpg".format(num)) #training outputs
        cv2.imwrite(base_path, output_image)
    else:
        os.makedirs('val_inputs', exist_ok=True) 
        base_path = os.path.join('val_inputs',"im{}.jpg".format(num)) #validation inputs
        cv2.imwrite(base_path, input_image)

        os.makedirs('val_outputs', exist_ok=True) 
        base_path = os.path.join('val_outputs',"im{}.jpg".format(num)) #validation outputs
        cv2.imwrite(base_path, output_image)
    count += 1