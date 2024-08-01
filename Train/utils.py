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
        self.data = torch.load(os.path.join(self.path, 'seg_data_' + mode + '.pt'), weights_only=True)
        self.label = torch.load(os.path.join(self.path, 'seg_label_' + mode + '.pt'), weights_only=True)
        self.length = len(self.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


# 没用，不要看

# class ModifiedDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
#         self.trans = transforms.Normalize(mean=(0.5), std=(0.5))

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img, label = self.dataset[idx]
#         img -= img.mean()
#         img /= img.std()
#         # img = self.trans(img)

#         # 将 (x, y, w, h) 格式的边界框转换为 (x1, y1, x2, y2) 格式
#         label[:, 2] = label[:, 0] + label[:, 2]
#         label[:, 3] = label[:, 1] + label[:, 3]

#         # 仅保留包含单词的边界框
#         indices = label.sum(dim=-1) > 0
#         label = label[indices]

#         # 制造classifier的标签
#         temp = torch.ones(len(label), dtype=torch.long)
#         label = {'boxes': label, 'labels': temp}
        
#         return img, label


# from torch.utils.data.dataloader import default_collate

# def collate_fn(batch):
#     images = [item[0] for item in batch]
#     targets = [item[1] for item in batch]

#     # 使用默认的 collate 处理图片（因为图片大小相同）
#     images = default_collate(images)
    
#     # 不尝试合并 targets，因为它们包含不同数量的边界框
#     # 直接作为列表返回
#     return images, targets


# class SegDatasetNew(Dataset):
#     def __init__(self, name, mode):
#         super(SegDatasetNew, self).__init__()
#         IAM_path = "/root/autodl-tmp/APS360_Project/Datasets/IAM_Processed"
#         CVL_path = "/root/autodl-tmp/APS360_Project/Datasets/CVL_Processed"
#         if name == 'IAM':
#             self.path = IAM_path
#         elif name == 'CVL':
#             self.path = CVL_path
#         else:
#             raise ValueError("Invalid dataset name")
#         self.name = name
#         self.mode = mode
#         self.data = torch.load(f"{self.path}/seg_data_{mode}.pt", weights_only=True)
#         self.target = torch.load(f"{self.path}/seg_label_{mode}_new.pt", weights_only=False)
#         self.length = len(self.data)
    
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, idx):
#         # make sure that the image is in 0-1 range
#         image = self.data[idx]
#         image -= image.min()
#         image /= image.max()
#         target = self.target['boxes'][idx]
#         labels = self.target['labels'][idx]
#         return image, {'boxes': target, 'labels': labels}

class RecDataset(Dataset):
    def __init__(self, name, mode, transform=None):
        """
        name: name of the dataset, IAM or CVL
        mode: train, val or test
        transform: default None, if not None, it should be a function
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
        self.transform = transform
        self.data = torch.load(os.path.join(self.path, 'rec_data_' + mode + '.pt'), weights_only=True)
        self.label = torch.load(os.path.join(self.path, 'rec_label_' + mode + '.pt'), weights_only=True)
        self.length = len(self.data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        img = data
        # print(img.min().item(), img.max().item(), img.mean().item(), img.std().item())
        data -= data.min()
        data /= data.max() / 2
        data -= 1
        if self.transform:
            # 如果有transform，那么就是将图片转成了PIL格式又转了回来，这时候黑白就会颠倒。所以要再转回来
            data = -data
        return data, self.label[idx]


class CRNN_Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def update(self, pred, target, is_original):
        if is_original:
            pred = pred.argmax(dim=-1)
        correct = (pred == target).sum().item()
        total = len(target)
        self.correct += correct
        self.total += total
    
    def get_accuracy(self):
        return self.correct / self.total


class RecDataset_Augmentation(Dataset):
    def __init__(self, name, mode, transform=None):
        self.path = "/root/autodl-tmp/APS360_Project/Datasets/Data_augmentation/Final"
        self.name = name
        self.data = torch.load(os.path.join(self.path, 'rec_data_' + mode + '.pt'))
        self.label = torch.load(os.path.join(self.path, 'rec_label_' + mode + '.pt'))

        if isinstance(self.data, list):
            self.data = self._convert_to_tensor(self.data)
        
        if isinstance(self.label, list):
            self.label = self._convert_labels_to_tensor(self.label)
        
        self.length = len(self.data)
        self.transform = transform

    def _convert_to_tensor(self, data):
        if isinstance(data, list):
            return torch.stack([self._convert_to_tensor(item) for item in data])
        return torch.tensor(data)
    
    def _convert_labels_to_tensor(self, labels):
        flat_labels = [word for sublist in labels for word in sublist]
        unique_labels = sorted(set(flat_labels))
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}
        index_labels = [[self.label_to_index[word] for word in line] for line in labels]
        return torch.tensor(index_labels)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

if __name__ == '__main__':
    dataset = RecDataset('IAM', 'train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (data, label) in enumerate(dataloader):
        print(data.min(), data.mean(), data.max(), data.dtype)
        print(data.shape)
        print(label.shape)
        break

# if __name__ == '__main__':
#     dataset = RecDataset_Augmentation('Data_Augmentation', 'train', transform=None)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#     for i, (data,label) in enumerate(dataloader):
#         print(data.min(), data.mean, data.max())
#         print(data.shape)
#         print(label.shape)
#         break

#     torch.set_printoptions(linewidth=200, precision=4, edgeitems=200, sci_mode=False)
#     print(label)