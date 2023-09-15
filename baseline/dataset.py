import pickle
import sys

import matplotlib.pyplot as plt
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from skimage import draw
from skimage import io
from config import *


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """
    Create an image mask from polygon coordinates
    """
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[fill_row_coords, fill_col_coords] = True
    mask = torch.tensor(mask.astype(np.uint8))
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor()])
    mask = transform(mask)
    return mask


def process_annotations(y_annotations):
    mask = torch.zeros(1, image_size, image_size)

    # Sum individual polygon masks to create one mask comprising of all individual polygons
    for poly in y_annotations:
        mask += poly2mask(poly[:, 1], poly[:, 0], (450, 600))
    ind = mask > 0.
    mask[ind] = 1.

    return mask


class MelanomaCharacteristicsDataset(Dataset):
    def __init__(self, root_dir, metadata, index=None, transform=None):
        self.root_dir = root_dir
        self.metadata = metadata
        if index is not None:
            self.metadata = self.metadata.loc[index]
        # self.y = self.metadata[dx_class_label].values.flatten()#.astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        img_path = os.path.join(self.root_dir, self.metadata.iloc[index]['image'])

        image = io.imread(img_path)
        y_dx = torch.tensor(self.metadata.iloc[index][dx_class_label]).float()
        image_name = self.metadata.iloc[index]['image']

        if self.transform:
            image = self.transform(image=image)['image']

        return image, (y_dx, image_name)

