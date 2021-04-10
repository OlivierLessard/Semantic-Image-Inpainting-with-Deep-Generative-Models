import torchvision
import torch


def get_data():
    # todo :calebA_data dataset
    # calebA_data = torchvision.datasets.CelebA(
    #     root="/datasets/celebA",
    #     split='all',
    #     # target_type
    #     transform=torchvision.transforms.CenterCrop(64),
    #     download=True
    # )
    # 2000 for tests

    svhn_train_data = torchvision.datasets.SVHN(
        root="Datasets\SVHN",
        split='train',
        # target_type
        transform=torchvision.transforms.Resize(size=[64, 64]),  # use svhn_train_data[index] to access the data with transformation, not svhn_train_data.data[index]
        download=True
    )

    svhn_test_data = torchvision.datasets.SVHN(
        root="Datasets\SVHN",
        split='test',
        # target_type
        transform=torchvision.transforms.Resize(size=[64, 64]),
        download=True
    )

    # todo : stanford cars dataset

    return svhn_train_data, svhn_test_data


def get_dataloaders(svhn_train_data, svhn_test_data, batch_size):
    # todo : calebA_data_loader
    # calebA_data_loader = torch.utils.data.DataLoader(calebA_data,
    #                                                  batch_size=16,
    #                                                  shuffle=True,
    #                                                  num_workers=4)

    svhn_train_data_loader = torch.utils.data.DataLoader(svhn_train_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4)
    svhn_test_data_loader = torch.utils.data.DataLoader(svhn_test_data,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4)

    # todo : stanford cars loader

    return svhn_train_data_loader, svhn_test_data_loader


if __name__ == '__main__':
    svhn_train_data, svhn_test_data = get_data()
    svhn_train_data_loader, svhn_test_data_loader = get_dataloaders(svhn_train_data, svhn_test_data)
