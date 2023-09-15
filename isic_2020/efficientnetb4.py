import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim
import cv2
cv2.setNumThreads(1)
import wandb
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.utils.data.sampler import WeightedRandomSampler
from dataset import *
import albumentations
from albumentations.pytorch import ToTensorV2
from captum.attr import LayerAttribution, LayerGradCam
from torchmetrics import Accuracy, AUROC, Recall, Specificity
from sklearn import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
num_workers = 16



def get_transforms(image_size, full=False):

    if full:
        transforms_train = albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ColorJitter(p=0.5),
            albumentations.OneOf([
                albumentations.MotionBlur(blur_limit=5),
                albumentations.MedianBlur(blur_limit=5),
                albumentations.GaussianBlur(blur_limit=(3, 5)),
                albumentations.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),
            albumentations.OneOf([
                albumentations.OpticalDistortion(distort_limit=1.0),
                albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                albumentations.ElasticTransform(alpha=3),
            ], p=0.7),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.3),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
            ToTensorV2()
        ])
    else:
        transforms_train = albumentations.Compose([
            albumentations.Transpose(p=0.2),
            albumentations.VerticalFlip(p=0.2),
            albumentations.HorizontalFlip(p=0.2),
            albumentations.ColorJitter(p=0.5),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.3),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
            ToTensorV2()
        ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2()
    ])
    return transforms_train, transforms_val


class EfficientNetB4(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, data_dir='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.data_dir = data_dir
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.threshold = 0.5
        self.train_set, self.val_set, self.test_set = None, None, None

        self.loss = nn.BCELoss()
        self.labels = dx_class_label
        self.num_classes = len(dx_class_label)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size, full=True)
        
        
        model = EfficientNet.from_pretrained('efficientnet-b4')
        model._fc =  nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model._fc.in_features, self.num_classes)
        )
        self.base_model = model

        self.sigm = nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=False),
            'interval': 'epoch',
            'frequency': 1
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def forward(self, x):
        output = self.sigm(self.base_model(x))

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, image_name = y
        output = self(x)
    
        loss = self.loss(output, y_dx)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, image_name = y
        with torch.no_grad():
            output = self(x)

        loss = self.loss(output, y_dx)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, image_name = y
        with torch.no_grad():
            output = self(x)
        loss = self.loss(output, y_dx)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, image_name = y
        with torch.no_grad():
            output = self(x)
        return output, y_dx, image_name

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':

            ##########################################################################################
            #### Read metadata and test lesions and assign test lesions from metadata to test_set ####
            ##########################################################################################
            test_img_ids = pd.read_csv(os.path.join(data_dir, 'metadata_testset.csv'))['image_id']
            test_lesion_ids = pd.read_csv(os.path.join(data_dir, 'metadata_testset.csv'))['lesion_id']

            metadata = pd.read_pickle(os.path.join(self.data_dir, 'metadata.pkl'))
            test_set = metadata[metadata['image_id'].isin(test_img_ids)] # now test set contains all images in test set by lesion
            
            #metadata.drop_duplicates(subset='lesion_id', keep='last', inplace=True)

            ######################################################################################################
            #### Subset metadata with lesions not in the test set, then split them into train and val lesions ####
            ######################################################################################################
            train = metadata[~metadata['lesion_id'].isin(test_lesion_ids)] # now train_set only contains lesions not in test set
            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )
            
            ##########################################################
            #### Select the train and val lesions from train data ####
            ##########################################################
            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(root_dir=self.data_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(root_dir=self.data_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(root_dir=self.data_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
            
    def train_dataloader(self):
        if self.weighted_sampling:
            y = self.train_set.metadata[dx_class_label].values.flatten().astype(int)
            counts = np.bincount(y)
            labels_weights = 1. / counts
            weights = labels_weights[y]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights))
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=num_workers)
        else:
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    