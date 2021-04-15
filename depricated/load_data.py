import torchvision
import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset
from torchvision import transforms
from skimage import transform
import numpy as np
import glob
import os
import PIL
from matplotlib import pyplot as plt


class OriginalImages(Dataset):
    def __init__(self, directory, image_size=(64, 64)):
        self.directory = directory
        self.images_filename = glob.glob(os.path.join(directory, "*.jpg"))
        self.image_size = image_size

    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):
        target_image = plt.imread(self.images_filename[idx])
        target_image = transform.resize(target_image, self.image_size)
        target_image = (target_image - np.min(target_image)) / np.max(target_image)  # values between 0 and 1
        return torch.FloatTensor(target_image).permute(2, 0, 1)  # (3, 64, 64)


def get_data(args):
    if args.dataset == "SVHN":
        train_data = torchvision.datasets.SVHN(
            root="Datasets\SVHN",
            split='train',
            # target_type
            transform=torchvision.transforms.Resize(size=[64, 64]),
            # use svhn_train_data[index] to access the data with transformation, not svhn_train_data.data[index]
            download=True
        )

        test_data = torchvision.datasets.SVHN(
            root="Datasets\SVHN",
            split='test',
            # target_type
            transform=torchvision.transforms.Resize(size=[64, 64]),
            download=True
        )

    # todo : stanford cars dataset
    elif args.dataset == "stanford_cars":
        print("stanford cars not implemented")

    return train_data, test_data


def get_dataloaders(train_data, test_data, args):
    batch_size = args.batch_size

    # todo : calebA_data_loader
    # calebA_data_loader = torch.utils.data.DataLoader(calebA_data,
    #                                                  batch_size=16,
    #                                                  shuffle=True,
    #                                                  num_workers=4)

    if args.dataset == "SVHN":
        train_data_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4)
        test_data_loader = torch.utils.data.DataLoader(test_data,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=4)

    # todo : stanford cars loader

    return train_data_loader, test_data_loader


if __name__ == '__main__':
    train_data, test_data = get_data()
    train_data_loader, test_data_loader = get_dataloaders(train_data, test_data)
