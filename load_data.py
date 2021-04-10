import torchvision
import torch


def get_data(args):
    # todo :calebA_data dataset
    if args.dataset == "calebA":
        print("calebA download not working")
        # calebA_data = torchvision.datasets.CelebA(
        #     root="/datasets/celebA",
        #     split='all',
        #     # target_type
        #     transform=torchvision.transforms.CenterCrop(64),
        #     download=True
        # )
        # 2000 for tests

    elif args.dataset == "SVHN":
        train_data = torchvision.datasets.SVHN(
            root="Datasets\SVHN",
            split='train',
            # target_type
            transform=torchvision.transforms.Resize(size=[64, 64]),  # use svhn_train_data[index] to access the data with transformation, not svhn_train_data.data[index]
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
