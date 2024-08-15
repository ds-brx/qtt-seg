
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import time
import random
from torchvision import models
from tqdm import tqdm
import pandas as pd
import os


class PascalVOCDataset(Dataset):
    def __init__(self, root_dir='/work/dlclarge2/dasb-Camvid/VOCdevkit/VOC2007', split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.masks_dir = os.path.join(root_dir, 'SegmentationClass')
        self.split_file = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        
        self.images = []
        self.masks = []
        
        with open(self.split_file, 'r') as file:
            file_names = file.read().splitlines()
        
        for file_name in file_names:
            img_path = os.path.join(self.images_dir, file_name + '.jpg')
            mask_path = os.path.join(self.masks_dir, file_name + '.png')
            
            self.images.append(img_path)
            self.masks.append(mask_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask


class Leaf_Dataset(Dataset):
    def __init__(self,root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        
        self.images = []
        self.masks = []

        for img_name in os.listdir(self.images_dir):
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name.replace(".jpg", ".png"))
            
            self.images.append(img_path)
            self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)    

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = mask.to(torch.long)
        return image, mask

class CamVidDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, class_dict_csv='/work/dlclarge2/dasb-Camvid/CamVid/class_dict.csv'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Directories for images and masks
        self.images_dir = os.path.join(root_dir, split)
        self.masks_dir = os.path.join(root_dir, f'{split}_labels')
        
        # List of images and masks
        self.images = []
        self.masks = []
        
        # Populate the lists with image and mask paths
        for img_name in os.listdir(self.images_dir):
            img_path = os.path.join(self.images_dir, img_name)
            mask_name = img_name.replace('.png', '_L.png')
            mask_path = os.path.join(self.masks_dir, mask_name)
            
            self.images.append(img_path)
            self.masks.append(mask_path)
        
        # Load the class mappings from CSV
        self.rgb_to_class = self.load_class_mappings(class_dict_csv)
        
    def load_class_mappings(self, csv_path):
        # Read the CSV file
        df = pd.read_csv(csv_path, sep='\t', header=None, names=['Class', 'R', 'G', 'B'])
        
        # Create a mapping from RGB tuples to class indices
        rgb_to_class = {}
        for idx, row in df.iterrows():
            rgb = (row['R'], row['G'], row['B'])
            rgb_to_class[rgb] = idx  # Use the row index as the class index
        
        return rgb_to_class

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Convert mask to class indices
        mask = np.array(mask)
        h, w, _ = mask.shape
        mask_indices = np.zeros((h, w))
        
        for rgb, class_idx in self.rgb_to_class.items():
            mask_indices[np.all(mask == rgb, axis=-1)] = class_idx
        
        # image = torch.tensor(image)
        # mask = torch.tensor(mask_indices, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask_indices)
        mask = mask.to(torch.long)
        
        return image, mask


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.masks_dir = os.path.join(root_dir, 'gtFine', split)

        
        self.images = []
        self.masks = []
        
        for city in os.listdir(self.images_dir):
            city_images_dir = os.path.join(self.images_dir, city)
            city_masks_dir = os.path.join(self.masks_dir, city)
            
            for img_name in os.listdir(city_images_dir):
                img_path = os.path.join(city_images_dir, img_name)
                mask_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                mask_path = os.path.join(city_masks_dir, mask_name)
                
                self.images.append(img_path)
                self.masks.append(mask_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask
