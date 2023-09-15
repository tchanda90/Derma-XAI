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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


#train_data_dir = "/home/kti01/Documents/My Files/Projects/Overlap/data/HAM10000/train"
#val_data_dir = "/home/kti01/Documents/My Files/Projects/Overlap/data/HAM10000/validation"
#est_data_dir = "./data/MSK"

data_dir = "../data/HAM10000"

model_save_dir = "./models"

weighted_sampling = True
learning_rate = 0.0001
batch_size = 16
num_epochs = 15
image_size = 224

save_attention_plots = False
dropout = 0.5

dx_class_label = ['benign_malignant']

seed = 42

seed_everything(seed)

