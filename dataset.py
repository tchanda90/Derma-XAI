import pickle
import sys
import json
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import draw
from skimage import io
from config import *


class MelanomaCharacteristicsDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, metadata, index=None, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata = metadata
        if index is not None:
            self.metadata = self.metadata.loc[index]
        self.y = self.metadata[char_class_labels].values.astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        image_id = self.metadata.iloc[index]['image_id'] 

        image = io.imread(os.path.join(self.img_dir, image_id + '.jpg'))
                
        y_dx = torch.tensor(self.metadata.iloc[index][dx_class_label]).float()
        y_char = torch.tensor(self.metadata.iloc[index][char_class_labels]).float()
        
        # Load the annotation json
        with open(os.path.join(self.annotations_dir, image_id+'.json'), 'r') as f:
            y_annotations = json.loads(json.load(f))
        
        # Store the feature masks in a list to pass to the augmentations function
        masks = []
        for char in char_class_labels:
            # Cast mask lists to np arrays. Assign zero valued masks to features not present in the image.
            y_annotations[char] = np.array(y_annotations[char]) if char in y_annotations else np.zeros((image_size, image_size))
            masks.append(y_annotations[char]) 

        if self.transform:
            transformed = self.transform(image=image, masks=masks)
            
            image, y_annotations = transformed['image'], transformed['masks']

        y_annotations = torch.tensor(y_annotations).float() 
                                    
        return image, (y_dx, y_char, y_annotations, image_id)



class HAM10000Dataset(Dataset):
    def __init__(self, root_dir, metadata, index=None, transform=None):
        self.root_dir = root_dir
        self.metadata = metadata
        if index is not None:
            self.metadata = self.metadata.loc[index]
        self.y = self.metadata[dx_class_label].values.flatten().astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.metadata.iloc[index]['image'])

        image = io.imread(img_path)
        y_dx = torch.tensor(self.metadata.iloc[index][dx_class_label]).float()
        y_char = torch.tensor(self.metadata.iloc[index][dx_class_label]).float()
        y_annotations = torch.zeros(1, image_size, image_size)
        image_name = self.metadata.iloc[index]['image']

        if self.transform:
            image = self.transform(image=image)['image']

        return image, (y_dx, y_char, y_annotations, image_name)


