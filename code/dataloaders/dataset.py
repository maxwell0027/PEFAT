# encoding: utf-8

# duplicate, only for debug
"""
Read images and corresponding labels.
"""

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import itertools
from torch.utils.data.sampler import Sampler
from torch.nn import functional as F


N_CLASSES = 9
CLASS_NAMES = {'ADI':0, 'BACK':1, 'LYM':2, 'STR':3, 'DEB':4, 'MUC':5, 'TUM':6, 'MUS':7, 'NORM':8}


def cutout(img, num_holes=8, length=28):
    """
    Args:
    img (Tensor): Tensor image of size (H, W, C). input is an image
    Returns:
    Tensor: Image with n_holes of dimension length x length cut out of it.
    """
    h = img.shape[1]
    w = img.shape[2]
    c = img.shape[0]
    mask = np.ones([h, w], np.float32)
    for _ in range(num_holes):
      y = np.random.randint(h)
      x = np.random.randint(w)
      y1 = np.clip(max(0, y - length // 2), 0, h)
      y2 = np.clip(max(0, y + length // 2), 0, h)
      x1 = np.clip(max(0, x - length // 2), 0, w)
      x2 = np.clip(max(0, x + length // 2), 0, w)
      mask[y1: y2, x1: x2] = 0
    mask = np.expand_dims(mask, 0)
    mask = torch.from_numpy(mask)
    
    mask = torch.cat((mask,mask,mask), dim=0)
    img = img * mask

    return img


class CheXpertDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, eval_train=False):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CheXpertDataset, self).__init__()
        fh = open(csv_file, 'r')
        self.img = []
        self.target = []
        self.eval_train = eval_train

        self.base_transform = transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])
                                          ])
        self.weak_transform = transforms.Compose([
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomRotation(10),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])
                                              ])
        
        for path in fh:
            self.img.append(path)
            cls_name = path.split('/')[-2]
            cls = F.one_hot(torch.tensor(CLASS_NAMES[cls_name]), num_classes = N_CLASSES)
            self.target.append(cls)

        self.transform = transform

        print('Total # images:{}'.format(len(self.img)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.img[index].strip()
        image_name = image_name[:6] + '/disk1' + image_name[6:]
        image = Image.open(image_name).convert('RGB')
        
        label = self.target[index]
        
        if self.eval_train == True:
            return [], index, self.base_transform(image), self.base_transform(image), label
        
        if self.transform is not None:
            image = self.transform(image)
        #print(label)
        return [], index, image, label

    def __len__(self):
        return len(self.img)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class TransformTwice:
    def __init__(self, transform):
        self.weak_transform = transforms.Compose([
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomRotation(10),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])
                                              ])
        self.color_transform = transforms.Compose([
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
                                              transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                              transforms.RandomGrayscale(p=0.3),
                                              transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])
                                              ])
        self.base_transform = transforms.Compose([
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])
                                          ])

    def __call__(self, inp):
        
        # note that weak and base for warm up and baseline training
        out1 = self.weak_transform(inp)
        out2 = self.base_transform(inp)
        
        '''
        # clolor and weak+cutout for others
        out1 = self.color_transform(inp)
        out2 = self.weak_transform(inp)         
        out2 = cutout(out2)
        '''
        return out1, out2

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
