import torchvision


def get_data():
    calebA_data = torchvision.datasets.CelebA(
        root="/datasets/celebA",
        split='all',
        # target_type
        transform=torchvision.transforms.CenterCrop(64),
        download=True
    )
    # 2000 for tests

    svhn_train_data = torchvision.datasets.SVHN(
        root="/datasets/SVHN",
        split='train',
        # target_type
        transform=torchvision.transforms.Resize((64, 64)),
        download=True
    )

    svhn_test_data = torchvision.datasets.SVHN(
        root="/datasets/SVHN",
        split='test',
        # target_type
        transform=torchvision.transforms.Resize((64, 64)),
        download=True
    )

    # todo : stanford cars dataset

    return calebA_data, svhn_train_data, svhn_test_data


if __name__ == '__main__':
    calebA_data, svhn_train_data, svhn_test_data = get_data()
