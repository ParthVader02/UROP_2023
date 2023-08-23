import cv2
import numpy as np
from skimage.util import random_noise
import random
import os

folder_dir = 'raw_data'
count = 1

for image in os.scandir(folder_dir):
    path = image.path
    if (path.endswith('.jpg')):
        img = cv2.imread(path)

        for i in range(10): #number of blurry images to generate from each sharp image

            blur_length = random.randint(10, 40) #change blur amount 
            angle = random.randint(0, 180) #change angle of blur
            thickness = random.randint(0, 1) #change thickness of blur

            psf = np.zeros((50, 50, 3)) #create empty kernel
            psf = cv2.ellipse(psf, 
                            (25, 25), # center
                            (blur_length, thickness), # axes -- 22 for blur length, 0 for thin PSF 
                            angle, # angle of motion in degrees
                            0, 360, # ful ellipse, not an arc
                            (1, 1, 1), # white color
                            thickness=-1) # filled
            
            psf /= psf[:,:,0].sum() # normalize by sum of one channel 
                            # since channels are processed independently

            imfilt = cv2.filter2D(img, -1, psf) # filter the image with the psf kernel

            noise_img = random_noise(imfilt, mode='gaussian', var=0.02**2) #CHECK THIS VALUE
            noise_img = np.array(255*noise_img, dtype = 'uint8')

            # resize the images for training on existing autoencoder
            new_size = (128, 128) # new_size=(width, height)
            input_image = cv2.resize(img, new_size)
            output_image = cv2.resize(noise_img, new_size)

            os.makedirs('sharp', exist_ok=True)
            base_path = os.path.join('sharp',"im{}.jpg".format(count))
            cv2.imwrite(base_path, input_image)

            os.makedirs('blurry', exist_ok=True)
            base_path = os.path.join('blurry',"im{}.jpg".format(count))
            cv2.imwrite(base_path, output_image)

            count += 1
        
        #every 10 images, capture a unalterated image pair 
        new_size = (128, 128) # new_size=(width, height)
        input_image = cv2.resize(img, new_size)
        output_image = cv2.resize(noise_img, new_size)

        os.makedirs('sharp', exist_ok=True)
        base_path = os.path.join('sharp',"im{}.jpg".format(count))
        cv2.imwrite(base_path, input_image)

        os.makedirs('blurry', exist_ok=True)
        base_path = os.path.join('blurry',"im{}.jpg".format(count))
        cv2.imwrite(base_path, output_image)
        count += 1

# img = cv2.imread("raw_data/im1.jpg")

# blur_length = random.randint(10, 40) #change blur amount 
# angle = random.randint(0, 180) #change angle of blur
# thickness = random.randint(0, 1) #change thickness of blur

# psf = np.zeros((50, 50, 3)) #create empty kernel
# psf = cv2.ellipse(psf, 
#             (25, 25), # center
#             (blur_length, thickness), # axes -- 22 for blur length, 0 for thin PSF 
#             angle, # angle of motion in degrees
#             0, 360, # ful ellipse, not an arc
#             (1, 1, 1), # white color
#             thickness=-1) # filled

# cv2.imshow("psf", psf)
# psf_save = np.array(255*psf, dtype = 'uint8')
# cv2.imwrite("psf_{}_{}_{}.jpg".format(blur_length, thickness, angle), psf_save)

# psf /= psf[:,:,0].sum() # normalize by sum of one channel 
#             # since channels are processed independently

# imfilt = cv2.filter2D(img, -1, psf) # filter the image with the psf kernel

# noise_img = random_noise(imfilt, mode='gaussian', var=0.02**2) #CHECK THIS VALUE
# noise_img = np.array(255*noise_img, dtype = 'uint8')

# cv2.imwrite("blurred_{}_{}_{}.jpg".format(blur_length, thickness, angle), noise_img)
# cv2.imshow("imfilt", noise_img)
# cv2.waitKey(0)