import os
import re

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

regex = r'.*ang(-*0\.\d).*'
pattern = re.compile(regex)


#taken from https://github.com/pytorch/vision/blob/fb83430f70bf9ba3923cba18956f6c80d163b6f2/torchvision/datasets/folder.py
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

#taken from https://github.com/pytorch/vision/blob/fb83430f70bf9ba3923cba18956f6c80d163b6f2/torchvision/datasets/folder.py
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class NavigationDataset(Dataset):
    def __init__(self,image_dirs, transform=None,descritize=False):
        self.all_data = []
        self.transform = transform
        self.descritize = descritize

        #this should be the last
        for image_dir in image_dirs:
            self.all_data += self.parse_image_dir(image_dir)

    def parse_image_dir(self,image_dir):
        all_data = []
        for filename in os.listdir(image_dir):
            if is_image_file(filename):
                image_filepath = os.path.join(image_dir,filename)
                ang_vel = filename.split("-")[2][3:]
                if len(ang_vel)==0:
                    value = float(filename.split("-")[3])
                    if not value  == 0:
                        ang_vel = -1.0 * value
                    else:
                        ang_vel = 0.0

                ang_vel=float(ang_vel)
                target = np.float32(ang_vel)

                if self.descritize:
                    if ang_vel > 0.02:
                        ang_vel_class=1#positive angular velocity
                    elif ang_vel < -0.02:
                        ang_vel_class=2#negative angular velocity
                    else:
                        ang_vel_class=0#zero angular velocity
                    target = ang_vel_class                    

                all_data.append([image_filepath,target])
        return all_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        img_path,target = self.all_data[idx]
        pil_image = pil_loader(img_path)
        img = pil_image

        if self.transform is not None:
            img = self.transform(img)

        return img,target