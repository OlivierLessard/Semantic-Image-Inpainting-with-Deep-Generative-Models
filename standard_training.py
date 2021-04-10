import argparse
from load_data import get_dataloaders, get_data, OriginalImages
from torch.utils.data import DataLoader
import models
from models import weights_init_normal
from torch import nn, optim
import torch
from matplotlib import pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SVHN")
    parser.add_argument("--mask-type", type=str, default="Center")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-path", type=str, default="./checkpoints/model.pth")
    parser.add_argument("--train-data-dir", type=str, default="./Datasets/CelebA/")

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--optim-steps", type=int, default=1500)
    parser.add_argument("--blending-steps", type=int, default=3000)
    parser.add_argument("--prior-weight", type=float, default=0.003)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=1)
    args = parser.parse_args()
    return args


def train_GAN(args):
    images_path = "./images/"
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    if not os.path.exists("./checkpoints/"):
        os.mkdir("./checkpoints/")

    # with CelebA
    dataset = OriginalImages(args.train_data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # show an image
    plt.imshow(dataset[0].permute(1, 2, 0))

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
            z = torch.FloatTensor(np.random.normal(0, 1, size=(real_images.shape[0], args.latent_dim))).cuda()
            fake_images = generator(z)
            real_images = real_images.cuda()
            real_labels = torch.FloatTensor(real_images.shape[0], 1).fill_(1.0).cuda()
            fake_labels = torch.FloatTensor(real_images.shape[0], 1).fill_(0.0).cuda()

            #  Train Generator
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(fake_images), real_labels)
            g_loss.backward()
            optimizer_G.step()
            total_G_loss += g_loss.cpu().detach().numpy()

            # train discriminator
            optimizer_G.zero_grad()
            D_on_real = discriminator(real_images)
            D_on_fake = discriminator(fake_images.detach())
            D_on_real_loss = adversarial_loss(D_on_real, real_labels)
            D_on_fake_loss = adversarial_loss(D_on_fake, fake_labels)
            D_loss = (D_on_real_loss + D_on_fake_loss)/2
            D_loss.backward()
            optimizer_D.step()
            total_D_loss += D_loss.cpu().detach().numpy()

            if epoch % args.sample_interval == 0 and i % (len(dataloader)/5) == 0:
                save_image(fake_images.data[0, 0], "images/{}_{}.png".format(
                    str(epoch).zfill(len(str(args.epochs))), str(i).zfill(len(str(len(dataloader))))), normalize=True)

        print("[Epoch {}/{}] \t[D loss: {:.3f}] \t[G loss: {:.3f}]".format(epoch, args.epochs, total_D_loss, total_G_loss))

        # save models during training
        torch.save({"epoch": epoch,
                    "state_dict_G": generator.state_dict(),
                    "state_dict_D": discriminator.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D": optimizer_D.state_dict()}
                   , args.model_path)

        if epoch % args.checkpoint_interval == 0:
            torch.save({"epoch": epoch,
                        "state_dict_G": generator.state_dict(),
                        "state_dict_D": discriminator.state_dict(),
                        "optimizer_G": optimizer_G.state_dict(),
                        "optimizer_D": optimizer_D.state_dict()
                        }, "checkpoints/{epoch}.pth".format(epoch=epoch))


if __name__ == "__main__":
    args = get_arguments()
    train_GAN(args)
