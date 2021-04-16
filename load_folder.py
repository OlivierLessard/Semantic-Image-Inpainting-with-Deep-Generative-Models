import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset
from torchvision import transforms
import glob, os
import numpy as np


def celeba_dataset_dataloader(args):
    # Create the dataset
    dataset = dset.ImageFolder(root=args.train_data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True)
    return dataset, dataloader


def svhn_dataset_dataloader(args):
    import torchvision
    train_data = torchvision.datasets.SVHN(root="Datasets\SVHN", split='train',
                                           transform=torchvision.transforms.Compose([
                                               transforms.Resize(args.image_size),
                                               transforms.CenterCrop(args.image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
                                           download=True
                                           )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                             shuffle=True)
    return train_data, dataloader
