# This file is used to store some useful functions for training

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SegDataset(Dataset):
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
        self.name = name
        self.mode = mode
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
        self.name = name
        self.mode = mode
        self.data = torch.load(os.path.join(self.path, 'rec_data_' + mode + '.pt'))
        self.label = torch.load(os.path.join(self.path, 'rec_label_' + mode + '.pt'))
        self.length = len(self.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class ModifiedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.trans = transforms.Normalize(mean=(0.5), std=(0.5))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img -= img.mean()
        img /= img.std()
        # img = self.trans(img)

        # 将 (x, y, w, h) 格式的边界框转换为 (x1, y1, x2, y2) 格式
        label[:, 2] = label[:, 0] + label[:, 2]
        label[:, 3] = label[:, 1] + label[:, 3]

        # 仅保留包含单词的边界框
        indices = label.sum(dim=-1) > 0
        label = label[indices]

        # 制造classifier的标签
        temp = torch.ones(len(label), dtype=torch.long)
        label = {'boxes': label, 'labels': temp}
        
        return img, label


from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 使用默认的 collate 处理图片（因为图片大小相同）
    images = default_collate(images)
    
    # 不尝试合并 targets，因为它们包含不同数量的边界框
    # 直接作为列表返回
    return images, targets


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = SegDataset('IAM', 'train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for data, label in dataloader:
        print(data.shape)
        print(label.shape)
        break