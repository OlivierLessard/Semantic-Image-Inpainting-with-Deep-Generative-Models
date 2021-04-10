import argparse
from load_data import get_dataloaders, get_data, OriginalImages
from torch.utils.data import DataLoader
import models
from models import weights_init_normal
from torch import nn, optim
import torch


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SVHN")
    parser.add_argument("--mask-type", type=str, default="Center")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gan-path", type=str, default="./checkpoints/model.pth")
    parser.add_argument("--train-data-dir", type=str, default="./Datasets/CelebA/")
    parser.add_argument("--eval-only", action='store_true', default=False)
    parser.add_argument("--test-only", action='store_true', default=False)

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--optim-steps", type=int, default=1500)
    parser.add_argument("--blending-steps", type=int, default=3000)
    parser.add_argument("--prior-weight", type=float, default=0.003)
    parser.add_argument("--window-size", type=int, default=25)
    args = parser.parse_args()
    return args


def inpaint(args):
    # with CelebA
    dataset = OriginalImages(args.train_data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # with SVHN
    # train_data, test_data = get_data(args)
    # train_data_loader, test_data_loader = get_dataloaders(train_data, test_data, args)

    # Define model
    generator = models.Generator(args).cuda()
    generator.apply(weights_init_normal)
    discriminator = models.Discriminator(args).cuda()
    discriminator.apply(weights_init_normal)

    # optimizer, loss
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters())
    optimizer_D = optim.Adam(discriminator.parameters())


    # train loop
    for epoch in range(args.epochs):
        total_G_loss = 0.0
        total_D_loss = 0.0

        for i, real_images in enumerate(dataloader):
            real_labels = torch.FloatTensor(real_images.shape[0], 1).fill_(1.0).cuda()
            fake_labels = torch.FloatTensor(real_images.shape[0], 1).fill_(0.0).cuda()


if __name__ == "__main__":
    args = get_arguments()
    inpaint(args)