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


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        num = targets.shape[0]
        inputs = inputs.reshape(num, -1)
        targets = targets.reshape(num, -1)

        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)

        dice = dice.sum() / num

        return 1 - dice


class MelanomaClassifier(pl.LightningModule):
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
        self.img_count = 1000
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.threshold = 0.5

        self.train_set, self.val_set, self.test_set = None, None, None

        self.loss = nn.BCELoss()

        self.labels = dx_class_label
        num_classes = len(dx_class_label)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)
        """
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Sequential(
            
            nn.Linear(in_features=resnet.fc.in_features, out_features=50),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=num_classes)
            nn.Dropout(p=0.5),
            nn.Linear(in_features=resnet.fc.in_features, out_features=num_classes)

        )
        """
        
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=resnet.fc.in_features, out_features=1),
            #nn.ReLU(),
            #nn.Linear(in_features=50, out_features=num_classes)
        )
        
        self.base_model = resnet

        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4)

        self.sigm = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=1)
        self.auroc = AUROC(num_classes=None, average='macro')
        self.sensitivity = Recall(threshold=0.5, average='micro')
        self.specificity = Specificity(threshold=0.5, average='micro')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def forward(self, x):
        output = self.sigm(self.base_model(x))

        return output

    def on_train_start(self):

        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, image_name = y
        output = self(x)
    
        loss = self.loss(output, y_dx)

        self.log("train/loss", loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, image_name = y
        with torch.no_grad():
            output = self(x)

        loss = self.loss(output, y_dx)

        self.log("val/loss", loss, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, image_name = y
        with torch.no_grad():
            output = self(x)
            attr = self.layer_gc.attribute(x, 0, relu_attributions=False)
            attributions = LayerAttribution.interpolate(attr, (image_size, image_size), interpolate_mode='bilinear')

        if save_attention_plots:
            for i in range(len(y_dx)):

                attr = attributions[i].detach().cpu().numpy()
                x_ = self.inverse_normalize(x[i]).detach().cpu().numpy()
                y_ = y_dx[i]
                out_ = output[i]
                
                if out_ >= 0.5:
                    target = 'mel'
                else:
                    target = 'nv'
                    attr = attr*-1 # Negative (nevus) attributions will become positive attributions                    

                plt.figure(figsize=(18, 12))
                
                plt.imshow(np.transpose(x_, (1, 2, 0)))
                plt.imshow(np.transpose(attr, (1, 2, 0)), alpha=0.3, cmap='jet')
                plt.colorbar()
                plt.savefig(os.path.join('plots/melanoma_classifier', target, image_name[i]))
                plt.close('all')
                
        loss = self.loss(output, y_dx)
        self.accuracy(output.flatten(), y_dx.int().flatten())
        self.auroc(output.flatten(), y_dx.int().flatten())
        self.sensitivity(output.flatten(), y_dx.int().flatten())
        self.specificity(output.flatten(), y_dx.int().flatten())
        
    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, image_name = y
        with torch.no_grad():
            output = self(x)
            attr = self.layer_gc.attribute(x, 0, relu_attributions=False)
            attributions = LayerAttribution.interpolate(attr, (image_size, image_size), interpolate_mode='bilinear')
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
            
            """
            train_set = metadata[~metadata['lesion_id'].isin(test_lesion_ids)]
            train_set, test_set = train_test_split(metadata, test_size=0.15, stratify=metadata[char_class_labels].values)
            train_set, val_set = train_test_split(train_set, test_size=0.18, stratify=train_set[char_class_labels].values)
            
            
            metadata = pd.read_pickle(os.path.join(self.data_dir, 'metadata.pkl'))
            test_imgs = pd.read_csv("data/HAM10000/0_ood_ham_loc_palms_soles")['image_id'].values
            train_set = metadata[~metadata['image_id'].isin(test_imgs)]
            test_set = metadata[metadata['image_id'].isin(test_imgs)]
            train_set, val_set = train_test_split(train_set, test_size=0.18, stratify=train_set[char_class_labels].values)
            
            #metadata = pd.read_pickle(os.path.join(self.data_dir, 'metadata.pkl'))
            #train_set, val_set = train_test_split(metadata, test_size=0.18, stratify=metadata[char_class_labels].values)
            #test_set = pd.read_csv(os.path.join(self.test_data_dir, 'metadata.csv')).dropna()
            """

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
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=1)
        else:
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        if False:# self.weighted_sampling:
            y = self.val_set.metadata[dx_class_label].values.flatten().astype(int)
            counts = np.bincount(y)
            labels_weights = 1. / counts
            weights = labels_weights[y]
            sampler = WeightedRandomSampler(weights, len(weights))
            return DataLoader(self.val_set, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=16)
        else:
            return DataLoader(self.val_set, batch_size=batch_size, shuffle=False, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=16)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=16)


    