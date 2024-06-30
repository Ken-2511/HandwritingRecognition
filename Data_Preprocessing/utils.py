# This file is used to store some useful functions for training

import os
import torch
from torch.utils.data import Dataset, DataLoader


class SegDataset(Dataset):
    def __init__(self, name, mode):
        """
        name: name of the dataset, IAM or CVL
        mode: train, val or test
        """
        CVL_path = '/root/autodl-tmp/APS360_Project/Datasets/CVL_Processed'
        IAM_path = '/root/autodl-tmp/APS360_Project/Datasets/IAM_Processed'
        self.name = name
        if name == 'IAM':
            self.path = IAM_path
        elif name == 'CVL':
            self.path = CVL_path
        else:
            raise ValueError('Invalid dataset name')
        if mode not in ['train', 'val', 'test']:
            raise ValueError('Invalid mode')
        self.data = torch.load(os.path.join(self.path, 'seg_data_' + mode + '.pt'))
        self.label = torch.load(os.path.join(self.path, 'seg_label_' + mode + '.pt'))
        self.length = len(self.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class RecDataset(Dataset):
    def __init__(self, name, mode):
        """
        name: name of the dataset, IAM or CVL
        mode: train, val or test
        """
        CVL_path = '/root/autodl-tmp/APS360_Project/Datasets/CVL_Processed'
        IAM_path = '/root/autodl-tmp/APS360_Project/Datasets/IAM_Processed'
        if name == 'IAM':
            self.path = IAM_path
        elif name == 'CVL':
            self.path = CVL_path
        else:
            raise ValueError('Invalid dataset name')
        if mode not in ['train', 'val', 'test']:
            raise ValueError('Invalid mode')
        self.data = torch.load(os.path.join(self.path, 'rec_data_' + mode + '.pt'))
        self.label = torch.load(os.path.join(self.path, 'rec_label_' + mode + '.pt'))
        self.length = len(self.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

