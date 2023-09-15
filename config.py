import random
import os
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = False

img_dir = "/home/caduser/Tirtha/overlap/data/HAM10000/HAM10000"
annotations_dir = "/home/caduser/Tirtha/overlap/data/ground_truth/annotations_gt"
metadata_file = "/home/caduser/Tirtha/overlap/data/ground_truth/metadata_gt.csv"
#metadata_file = "metadata_undersampled.csv"


model_save_dir = "./models"

weighted_sampling = False
save_attention_plots = False


num_epochs = 30
learning_rate = 0.0001
batch_size = 32
image_size = 224

attention_weight = 10
char_weight = 1
dropout = 0.4

mel_class_labels = ['TRBL', 'BDG', 'WLSA', 'ESA', 'GP', 'PV', 'PRL']
nev_class_labels = ['APC', 'MS', 'OPC']
char_class_labels = mel_class_labels+nev_class_labels 
pos_weight = torch.tensor([2, 2, 2.2, 2.2, 2, 2.5, 2.5, 0.5, 0.5, 0.9])


char_class_labels_pred = [label+'_pred' for label in char_class_labels]
mel_class_labels_pred = [label+'_pred' for label in mel_class_labels]
nev_class_labels_pred = [label+'_pred' for label in nev_class_labels]
char_class_labels_score = [label+'_score' for label in char_class_labels]

annotation_labels = [label+'_annotation' for label in char_class_labels]

dx_class_label = ['benign_malignant']

seed = 42

seed_everything(seed)

