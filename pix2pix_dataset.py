import datetime
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import Dataset

import PIL
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

root_dir = "/home/sju07144/Capture_231113" # current directory is "/home/sju07144/pix2pix"
root_512_dir = "/home/hoyeon/Capture_512"
# root_dir = "..\\resources\\IBL_rendered_examples"

modes = [
  'Albedo',
  'AO',
  'Metallic',
  'Metallic-Roughness',
  'NormalMap',
  'Normal',
  'Roughness'
]

class ABODataset(Dataset):
    def __init__(self, root=root_dir, train=True, mode=0):
        self.real_image_paths = []
        self.input_image_paths = []
        
        self.mask_image_paths = []
        
        self.train = train
        
        self.all_subdirs = [str(i) for i in range(10)]
        for one in range(65, 91):
            self.all_subdirs.append(chr(one)) 
        self.dataset_subdirs = []
        
        '''
        Train sub_directories: 0-9A-W
        Test sub_directories: X-Z
        Train Test ratio: 90.4%/9.6%
        '''
        
        if self.train:
            for i in range(len(self.all_subdirs) - 3):
                self.dataset_subdirs.append(os.path.join(root, self.all_subdirs[i]))
        else:
            for i in range(-3, 0):
                self.dataset_subdirs.append(os.path.join(root, self.all_subdirs[i]))
                
        for dataset_dir in self.dataset_subdirs:
            for (root, dirs, files) in os.walk(dataset_dir):
                if len(dirs) > 0:
                    for dir_name in dirs:
                        if dir_name.find('.glb'):
                            model_dir = os.path.join(dataset_dir, dir_name)

                        for (root, dirs, files) in os.walk(model_dir):
                            mode_temp_image_paths = []
                            mask_temp_image_paths = []
                            if len(files) > 0:
                                for file_name in files:
                                    if file_name.find(modes[mode]) != -1:
                                        mode_temp_image_paths.append(os.path.join(model_dir, file_name))

                                    elif file_name.find('Mask') != -1:
                                        mask_temp_image_paths.append(os.path.join(model_dir, file_name))

                                    elif file_name.find('_IBL_') != -1 and file_name.find('_IBL_IBR_') == -1: # HDR -> _IBL_
                                        self.input_image_paths.append(os.path.join(model_dir, file_name))

                                for _ in range(6):
                                    for image_path in mode_temp_image_paths:
                                        self.real_image_paths.append(image_path)    
                                    for image_path in mask_temp_image_paths:
                                        self.mask_image_paths.append(image_path)  
                                        
    def __len__(self): 
        return len(self.input_image_paths)
    
    def __getitem__(self, index):
        real_image = read_image(self.real_image_paths[index])
        input_image = read_image(self.input_image_paths[index])
        mask_image = read_image(self.mask_image_paths[index])
        
        alpha_channel_mask = mask_image[0:1, :, :]
        real_image = torch.cat([real_image[:3, :, :], alpha_channel_mask], dim=0)
        input_image = torch.cat([input_image[:3, :, :], alpha_channel_mask], dim=0)
        
        real_image = real_image.float()
        input_image = input_image.float()
         
        if self.train:
            transform = transforms.Compose([
                transforms.Resize((552, 552), 
                                  interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((512, 512), 
                                  interpolation=transforms.InterpolationMode.NEAREST)
            ])
            
        real_image = transform(real_image)  
        input_image = transform(input_image)
        
        real_image = real_image / 127.5 - 1.0
        input_image = input_image / 127.5 - 1.0
        
        return input_image, real_image