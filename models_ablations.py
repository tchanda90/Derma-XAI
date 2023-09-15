import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(1)
import wandb
import pytorch_lightning as pl
import albumentations
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from albumentations.pytorch import ToTensorV2
from captum.attr import LayerAttribution, LayerGradCam
from torchmetrics import Accuracy, AUROC, Recall, Specificity
from sklearn import metrics
from sklearn.model_selection import train_test_split
from dataset import *
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
            albumentations.Resize(image_size, image_size),
            albumentations.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.3),
            albumentations.Normalize(),
            ToTensorV2()
        ])
    else:
        transforms_train = albumentations.Compose([
            albumentations.Transpose(p=0.2),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ColorJitter(p=0.5),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            #albumentations.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.3),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
            ToTensorV2()
        ])

    transforms_test = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(),
        ToTensorV2()
    ])
    return transforms_train, transforms_test


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
    

    
    
    

class Resnet50(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.dataset = []
        self.masks = []

        self.train_set, self.val_set, self.test_set = None, None, None

        self.lossC = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossA = DiceLoss() # nn.MSELoss()
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=resnet.fc.in_features, out_features=self.num_classes)
            #nn.ReLU(),
            #nn.Dropout(p=dropout),
            #nn.Linear(in_features=50, out_features=self.num_classes)
        )
        self.base_model = resnet
    
        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4[-1])

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = 'base_model.layer4'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        output, attributions = self(x)
        
                
        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight
        
        self.lossA_list.append(lossA.item())
        self.lossC_list.append(lossC.item())

        
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/lossA", lossA, on_epoch=True, on_step=False)
        self.log("train/lossC", lossC, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())


    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)
        
        return output, attributions, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            
            test_set = metadata[metadata['split'] == 'test']
            train = metadata[metadata['split'] == 'train']
            
            # Drop lesion Ids from train set that are also in test set
            train = train[~train['lesion_id'].isin(test_set['lesion_id'])]
            
            # Drop rows where all labels are 0
            #train = train.loc[(train[char_class_labels] != 0).any(axis=1)]


            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )
            
            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)


    

    
class Resnet101(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.dataset = []
        self.masks = []

        self.train_set, self.val_set, self.test_set = None, None, None

        self.lossC = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossA = DiceLoss() # nn.MSELoss()
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        resnet = models.resnet101(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features=resnet.fc.in_features, out_features=self.num_classes)
            #nn.ReLU(),
            #nn.Dropout(p=dropout),
            #nn.Linear(in_features=50, out_features=self.num_classes)
        )
        self.base_model = resnet
    
        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4[-1])

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = 'base_model.layer4'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        output, attributions = self(x)
        
                
        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight
        
        self.lossA_list.append(lossA.item())
        self.lossC_list.append(lossC.item())

        
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/lossA", lossA, on_epoch=True, on_step=False)
        self.log("train/lossC", lossC, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())


    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)
        
        return output, attributions, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            
            test_set = metadata[metadata['split'] == 'test']
            train = metadata[metadata['split'] == 'train']
            
            # Drop lesion Ids from train set that are also in test set
            train = train[~train['lesion_id'].isin(test_set['lesion_id'])]
            
            # Drop rows where all labels are 0
            #train = train.loc[(train[char_class_labels] != 0).any(axis=1)]


            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )
            
            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)


    


    


class Resnet34(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.dataset = []
        self.masks = []

        self.train_set, self.val_set, self.test_set = None, None, None

        self.lossC = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossA = DiceLoss() # nn.MSELoss()
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        resnet = models.resnet34(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=resnet.fc.in_features, out_features=self.num_classes)
            #nn.ReLU(),
            #nn.Dropout(p=dropout),
            #nn.Linear(in_features=50, out_features=self.num_classes)
        )
        self.base_model = resnet
    
        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4[-1])

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = 'base_model.layer4'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        output, attributions = self(x)
        
                
        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight
        
        self.lossA_list.append(lossA.item())
        self.lossC_list.append(lossC.item())

        
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/lossA", lossA, on_epoch=True, on_step=False)
        self.log("train/lossC", lossC, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())


    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)
        
        return output, attributions, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            
            test_set = metadata[metadata['split'] == 'test']
            train = metadata[metadata['split'] == 'train']
            
            # Drop lesion Ids from train set that are also in test set
            train = train[~train['lesion_id'].isin(test_set['lesion_id'])]
            
            # Drop rows where all labels are 0
            #train = train.loc[(train[char_class_labels] != 0).any(axis=1)]


            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )
            
            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    
    
    
    

class Resnet18(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.dataset = []
        self.masks = []

        self.train_set, self.val_set, self.test_set = None, None, None

        self.lossC = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossA = DiceLoss() # nn.MSELoss()
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=resnet.fc.in_features, out_features=self.num_classes)
            #nn.ReLU(),
            #nn.Dropout(p=dropout),
            #nn.Linear(in_features=50, out_features=self.num_classes)
        )
        self.base_model = resnet
    
        self.layer_gc = LayerGradCam(self.base_model, self.base_model.layer4[-1])

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = 'base_model.layer4'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        output, attributions = self(x)
        
                
        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight
        
        self.lossA_list.append(lossA.item())
        self.lossC_list.append(lossC.item())

        
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/lossA", lossA, on_epoch=True, on_step=False)
        self.log("train/lossC", lossC, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())


    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)
        
        return output, attributions, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            
            test_set = metadata[metadata['split'] == 'test']
            train = metadata[metadata['split'] == 'train']
            
            # Drop lesion Ids from train set that are also in test set
            train = train[~train['lesion_id'].isin(test_set['lesion_id'])]
            
            # Drop rows where all labels are 0
            #train = train.loc[(train[char_class_labels] != 0).any(axis=1)]


            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )
            
            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class Densenet121(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.dataset = []
        self.masks = []

        self.train_set, self.val_set, self.test_set = None, None, None

        self.lossC = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossA = DiceLoss() # nn.MSELoss()
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)
        
        densenet = models.densenet121(pretrained=True)
        densenet.classifier = torch.nn.Linear(densenet.classifier.in_features, self.num_classes)
                
        self.base_model = densenet
    
        self.layer_gc = LayerGradCam(self.base_model, self.base_model.features.denseblock4.denselayer16)

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = 'features.denseblock4.denselayer16'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.base_model.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        output, attributions = self(x)
        
                
        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight
        
        self.lossA_list.append(lossA.item())
        self.lossC_list.append(lossC.item())

        
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/lossA", lossA, on_epoch=True, on_step=False)
        self.log("train/lossC", lossC, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())


    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)
        
        return output, attributions, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            
            test_set = metadata[metadata['split'] == 'test']
            train = metadata[metadata['split'] == 'train']
            
            # Drop lesion Ids from train set that are also in test set
            train = train[~train['lesion_id'].isin(test_set['lesion_id'])]
            
            # Drop rows where all labels are 0
            #train = train.loc[(train[char_class_labels] != 0).any(axis=1)]


            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )
            
            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
    
    
    
    
    

    
    
    
    

    

class Densenet161(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.dataset = []
        self.masks = []

        self.train_set, self.val_set, self.test_set = None, None, None

        self.lossC = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossA = DiceLoss() # nn.MSELoss()
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)
        
        densenet = models.densenet161(pretrained=True)
        densenet.classifier = torch.nn.Linear(densenet.classifier.in_features, self.num_classes)
                
        self.base_model = densenet
    
        self.layer_gc = LayerGradCam(self.base_model, self.base_model.features.denseblock4.denselayer24)

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = 'features.denseblock4.denselayer24'
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.base_model.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        output, attributions = self(x)
        
                
        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight
        
        self.lossA_list.append(lossA.item())
        self.lossC_list.append(lossC.item())

        
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/lossA", lossA, on_epoch=True, on_step=False)
        self.log("train/lossC", lossC, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())


    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)
        
        return output, attributions, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            
            test_set = metadata[metadata['split'] == 'test']
            train = metadata[metadata['split'] == 'train']
            
            # Drop lesion Ids from train set that are also in test set
            train = train[~train['lesion_id'].isin(test_set['lesion_id'])]
            
            # Drop rows where all labels are 0
            #train = train.loc[(train[char_class_labels] != 0).any(axis=1)]


            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )
            
            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
    
    
    
    
    
    
    

class EfficientNetB1(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.dataset = []
        self.masks = []

        self.train_set, self.val_set, self.test_set = None, None, None

        self.lossC = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossA = DiceLoss() # nn.MSELoss()
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
        efficientnet._fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=efficientnet._fc.in_features, out_features=self.num_classes)
        )
        self.base_model = efficientnet
    
        self.layer_gc = LayerGradCam(self.base_model, self.base_model._conv_head)

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = '_conv_head' 
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.base_model.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        output, attributions = self(x)
        
                
        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight
        
        self.lossA_list.append(lossA.item())
        self.lossC_list.append(lossC.item())

        
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/lossA", lossA, on_epoch=True, on_step=False)
        self.log("train/lossC", lossC, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())


    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)
        
        return output, attributions, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            
            test_set = metadata[metadata['split'] == 'test']
            train = metadata[metadata['split'] == 'train']
            
            # Drop lesion Ids from train set that are also in test set
            train = train[~train['lesion_id'].isin(test_set['lesion_id'])]
            
            # Drop rows where all labels are 0
            #train = train.loc[(train[char_class_labels] != 0).any(axis=1)]


            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )
            
            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    
    
    
    
    

    
    
    

    
    

class EfficientNetB3(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, img_dir='./', annotations_dir='./', metadata_file='./', train_data_dir='./', val_data_dir='./',
                 test_data_dir='./', batch_size=32, weighted_sampling=False):

        super().__init__()
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.metadata_file = metadata_file
        
        self.weighted_sampling = weighted_sampling
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.attributions = []
        self.y_annotations = []
        self.y_char = []
        self.lossC_list = []
        self.lossA_list = []
        self.dataset = []
        self.masks = []

        self.train_set, self.val_set, self.test_set = None, None, None

        self.lossC = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lossA = DiceLoss() # nn.MSELoss()
        
        self.labels = char_class_labels
        self.num_classes = len(char_class_labels)

        self.train_transform, self.test_transform = get_transforms(image_size=image_size)

        efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        efficientnet._fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=efficientnet._fc.in_features, out_features=self.num_classes)
        )
        self.base_model = efficientnet
    
        self.layer_gc = LayerGradCam(self.base_model, self.base_model._conv_head)

        self.sigmoid = nn.Sigmoid()

        self.accuracy = Accuracy(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.auroc = AUROC(average='macro', num_classes=self.num_classes)
        self.sensitivity = Recall(threshold=0.5, average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(threshold=0.5, average='macro', num_classes=self.num_classes)
        
        self.target_layer = '_conv_head' 
        self.fmap = None
        self.grad = None
        self.handlers = []

        def save_fmaps(key):
            def hook(module, fmap_in, fmap_out):
                if isinstance(fmap_out, list):
                    fmap_out = fmap_out[-1]
                self.fmap = fmap_out

            return hook

        def save_grads(key):
            def hook(module, grad_in, grad_out):
                self.grad = grad_out[0]

            return hook

        target_module = dict(self.base_model.named_modules())[self.target_layer]
        self.handlers.append(
            target_module.register_forward_hook(save_fmaps(self.target_layer))
        )
        self.handlers.append(
            target_module.register_backward_hook(save_grads(self.target_layer))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    def attribute(self, logits, class_idx):
        self.zero_grad()
        one_hot = F.one_hot((torch.ones(logits.size(0)) * class_idx).long(), self.num_classes).to(self.device)
        logits.backward(gradient=one_hot, retain_graph=True)
        weights = F.adaptive_avg_pool2d(self.grad, 1)
        attr = torch.mul(self.fmap, weights).sum(dim=1, keepdim=True)
        #attr = self.sigmoid(F.relu(attr)) * 2 - 1
        attr = F.relu(attr)
        return attr

    def forward(self, x, zero_non_predicted=True):
        
        output = self.base_model(x)
                
        # Get attributions for each class
        #attributions = torch.cat([self.layer_gc.attribute(x, class_idx, relu_attributions=True) for class_idx in range(self.num_classes)], dim=1)
        attributions = torch.cat([self.attribute(output, class_idx) for class_idx in range(self.num_classes)], dim=1)
        # Upsample the attributions to match the image size
        attributions = F.interpolate(attributions, size=(image_size, image_size), mode='nearest')
        
        predicted_classes = torch.round(self.sigmoid(output))
        # Set attributions of non predicted classes to zero by multiplying with the rounded prediction: 1 or 0.
        if zero_non_predicted:
            for batch_idx in range(attributions.size(0)):
                for class_idx in range(self.num_classes):
                    attributions[batch_idx][class_idx] *= predicted_classes[batch_idx][class_idx]
        
        return output, attributions

    def on_train_start(self):

        self.log("hp/attention_weight", float(attention_weight))
        self.log("hp/lr", learning_rate)
        self.log("hp/batch_size", float(batch_size))
        self.log("hp/dropout", dropout)
        self.log("hp/weighted_sampling", float(weighted_sampling))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        output, attributions = self(x)
        
                
        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight
        
        self.lossA_list.append(lossA.item())
        self.lossC_list.append(lossC.item())

        
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log("train/lossA", lossA, on_epoch=True, on_step=False)
        self.log("train/lossC", lossC, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log("val/lossA", lossA, on_epoch=True, on_step=False)
        self.log("val/lossC", lossC, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)

        lossA = self.lossA(attributions, y_annotations)
        lossC = self.lossC(output, y_char)

        loss = lossC * char_weight + lossA * attention_weight

        self.accuracy(output.round(), y_char.int())
        self.auroc(output, y_char.int())
        self.sensitivity(output, y_char.int())
        self.specificity(output, y_char.int())


    def on_test_epoch_end(self) -> None:
        self.log("test/bal_acc", self.accuracy.compute())
        self.log("test/auroc", self.auroc.compute())
        self.log("test/sensitivity", self.sensitivity.compute())
        self.log("test/specificity", self.specificity.compute())


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_dx, y_char, y_annotations, image_name = y
        
        with torch.set_grad_enabled(True):
            output, attributions = self(x)
        
        return output, attributions, y_char, image_name, self.inverse_normalize(x)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        if stage == 'fit':
            metadata = pd.read_csv(metadata_file)#.drop_duplicates(subset='lesion_id', keep='last')
            
            test_set = metadata[metadata['split'] == 'test']
            train = metadata[metadata['split'] == 'train']
            
            # Drop lesion Ids from train set that are also in test set
            train = train[~train['lesion_id'].isin(test_set['lesion_id'])]
            
            # Drop rows where all labels are 0
            #train = train.loc[(train[char_class_labels] != 0).any(axis=1)]


            train_lesions, val_lesions = train_test_split(
                train.drop_duplicates('lesion_id')['lesion_id'], test_size=0.18, stratify=train.drop_duplicates('lesion_id')[dx_class_label],
                random_state=seed
            )
            
            train_set = train[train['lesion_id'].isin(train_lesions)]
            val_set = train[train['lesion_id'].isin(val_lesions)]

            self.train_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                            annotations_dir=self.annotations_dir,
                                                            metadata=train_set,
                                                            transform=self.train_transform)

            self.val_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                          annotations_dir=self.annotations_dir,
                                                          metadata=val_set,
                                                          transform=self.test_transform)

            self.test_set = MelanomaCharacteristicsDataset(img_dir=self.img_dir,
                                                           annotations_dir=self.annotations_dir,
                                                           metadata=test_set,
                                                           transform=self.test_transform)
    

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    