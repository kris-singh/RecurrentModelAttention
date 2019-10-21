#!/usr/bin/env python

import argparse
import os
from collections import defaultdict

import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset



class ModMnist(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        super(ModMnist, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def split_dataset(root, splits):
    # shuffle
    path = os.path.join(root, 'MNIST/processed/training.pt')
    dataset, targets = torch.load(path)
    idx = torch.randperm(len(dataset))
    dataset = dataset[idx]
    targets = targets[idx]
    train_idx = int(splits['train']*len(dataset))
    train_data = dataset[:train_idx]
    train_targets = targets[:train_idx]
    val_data = dataset[train_idx:]
    val_targets = targets[train_idx:]
    return train_data, train_targets, val_data, val_targets


def get_loader(cfg, type):
    kwargs = {'num_workers': cfg.SYSTEM.NUM_WORKERS, 'pin_memory': cfg.SYSTEM.PIN_MEMORY}
    splits = defaultdict()
    splits['train'] = 0.75
    train_data, train_target, val_data, val_target = split_dataset(cfg.TRAIN.ROOT, splits)

    if type=='train':
        train_dataset = ModMnist(train_data, train_target, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                                   shuffle=True, **kwargs)
        return train_loader

    elif type=='val':
        val_dataset = ModMnist(val_data, val_target, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=cfg.VAL.BATCH_SIZE,
                                                   shuffle=True, **kwargs)
        return val_loader

    elif type=='test':
        test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/Users/kris/torch/data', train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, **kwargs)
