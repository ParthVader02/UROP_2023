import cv2
import numpy as np
import pickle

from digit_interface import Digit

class DigitSensor():
    """
    Digit sensor class
    """
    def __init__(self, serialno: str = 'D20654', resolution: str = 'QVGA', framerate: int = 60):
        self.frame_count = 0
        self.serialno = serialno
        self.name = 'DIGIT'
        self.resolution = resolution
        self.framerate = framerate
        self.digit = None

    def __enter__(self):
        # Connect to Digit sensor
        self.digit = Digit(self.serialno, self.name)
        self.digit.connect()

        # Configure sensor resolution
        resolution = Digit.STREAMS[self.resolution]
        self.digit.set_resolution(resolution)

        # Configure sensor framerate
        framerate = Digit.STREAMS[self.resolution]["fps"]["{}fps".format(self.framerate)]
        self.digit.set_fps(framerate)

        return self
    
    def get_frame(self, transpose: bool = False):
        self.frame_count += 1
        return self.digit.get_frame(transpose)

    def disconnect(self):
        # Disconnect Digit sensor
        self.digit.disconnect()
    
    def __exit__(self, exception_type, exception_value, traceback):
        # Disconnect Digit sensor
        self.disconnect()
    
class DisplayImage():
    """
    Provides convenient functions
    for modifying and displaying images.
    """

    def __init__(self, window_name: str = 'DIGIT'):
        self.window_name = window_name
        return

    def __enter__(self):
        return self
    
    def show_image(
            self, 
            images: list[np.array],
            resize: str = 'pad',
            window_scale: float = 1.0,
            image_frame: bool = False,
            n_rows: int = 1):
        """
        Display list of images in a single window using OpenCV

        Parameters
        ----------
            `images`       - list of np.array instances to be displayed.\n
            `resize`       - either 'pad' to add black border or 'repeat' to scale image to correct resolution using nearest neighour.\n
            `window_scale` - global scaling of displayed image.\n
            `image_frame`  - add black border to displayed images.\n
            `n_rows`       - number of rows the images are displayed on: len(images)%n_rows must equal 0.
        """

        img_pad = []
        n_images = len(images)
        images_per_row = int(n_images/n_rows)

        if (n_images%n_rows != 0):
            raise ValueError("The number of images dispayed must be divisible by n_rows")
        
        img_rows = []
        for row in range(n_rows):
            img_pad = []
            for i in range(row*images_per_row, (1+row)*images_per_row):
                image = images[i]
                if (images[i].shape[0] < images[0].shape[0]) or (images[i].shape[1] < images[0].shape[1]):
                    if resize=='pad':
                        image = np.pad(
                            images[i], 
                            pad_width=[
                                (int((images[0].shape[0]-images[i].shape[0])/2), int((images[0].shape[0]-images[i].shape[0])/2)),
                                (int((images[0].shape[1]-images[i].shape[1])/2), int((images[0].shape[1]-images[i].shape[1])/2)),
                                (0, 0)], 
                            mode='constant')
                    elif resize=='repeat':
                        img_tmp = np.repeat(images[i], repeats=int(images[0].shape[0]/images[i].shape[0]), axis=0)
                        image = np.repeat(img_tmp, repeats=int(images[0].shape[0]/images[i].shape[0]), axis=1)
                    else:
                        raise ValueError('Invalid resize method')
                if image_frame==True:
                    image = np.pad(image, [(1,1),(1,1),(0,0)], mode='constant')
                img_pad.append(image)
            img_rows.append(np.concatenate(img_pad, axis=1))
        if n_rows>1:
            img = np.concatenate(img_rows, axis=0)
        else:
            img = img_rows[0]
        img = cv2.resize(img, (int(window_scale*img.shape[1]),int(window_scale*img.shape[0])))
        cv2.imshow(self.window_name, img)

        return

    def save_array(
            self, 
            images: list, 
            directory: str, 
            filename: str, 
            frame_number: int, 
            frame_skip: int = 4, 
            visualise: bool = False):
        """
        Saves current frame using pickle

        Parameters
        ----------
            `images`       - list of np.array instances to be displayed\n
            `directory`    - the directory to save images\n
            `filename`     - the filename of saved images\n
            `frame_number` - the current frame count\n
            `visualise`    - display the current image using cv2
        """

        img = np.concatenate(images, axis=1)
        if frame_number%frame_skip==0:
            if visualise:
                cv2.imshow(self.window_name, img)
            # cv2.imwrite('{}{}_{}.png'.format(directory,filename,str(frame_number).zfill(5)), img)
            with open('{}{}_{}.pickle'.format(directory,filename,str(frame_number).zfill(5)), 'wb') as handle:
                pickle.dump(img, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return
    
    def __exit__(
            self, 
            exception_type, 
            exception_value, 
            traceback):
            
        # Disconnect Digit sensor
        cv2.destroyAllWindows()
        return
import os
def example_capture_video():

    with (DigitSensor(serialno='D20652', resolution='QVGA', framerate='60') as digit):

        while True:
            frame = digit.get_frame()
            cv2.imshow('DIGIT Demo', frame)
            m = 0.5
            force =  m*9.81
            dir_path = '/home/parth/UROP_2023/force_images'
            # '27' is the escape key
            if cv2.waitKey(1)==27:
                # os.makedirs(dir_path, exist_ok=True)  
                # base_path = os.path.join(dir_path,"im{}.jpg".format(force)) #create path to save frame
                # cv2.imwrite(base_path, frame)
                break

if __name__=='__main__':
    example_capture_video()