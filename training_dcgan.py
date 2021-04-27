from dcgan import Generator
import argparse
from depricated import models
from depricated.models import weights_init_normal
from torch import nn, optim
import torch
from matplotlib import pyplot as plt
import numpy as np
import os, torchvision
from torchvision.utils import save_image
import random
from data import celeba_dataset_dataloader
from dcgan import weights_init
import torchvision.utils as vutils
from dcgan import Discriminator
from visualization import save_learning_curves
import matplotlib.animation as animation
from IPython.display import HTML


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SVHN")
    parser.add_argument("--mask-type", type=str, default="Center")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-path", type=str, default="./checkpoints_celebA_dcgan/dcgan.pth")
    parser.add_argument("--train-data-dir", type=str, default="./Datasets/CelebA/")

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--optim-steps", type=int, default=1500)
    parser.add_argument("--blending-steps", type=int, default=3000)
    parser.add_argument("--prior-weight", type=float, default=0.003)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=300)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--nc", type=int, default=3)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()
    return args


def create_folders():
    if not os.path.exists("checkpoints_celebA_dcgan/"):
        os.mkdir("checkpoints_celebA_dcgan/")
    if not os.path.exists("./eval/"):
        os.mkdir("./eval/")
    if not os.path.exists("./Output/"):
        os.mkdir("./Output/")


def train_GAN(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    print("device:", device)
    create_folders()

    # Set random seed for reproducibility
    seed = 999
    random.seed(seed)
    torch.manual_seed(seed)

    # get celebA dataset and dataloader
    dataset, dataloader = celeba_dataset_dataloader(args)
    print("len dataset: ", dataset.__len__())

    # Create the generator
    netG = Generator(args).to(device)
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(args).to(device)
    netD.apply(weights_init)

    # Print the model
    print(netG)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)

            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            if iters % args.checkpoint_interval == 0:
                torch.save({"iteration": iters,
                            "state_dict_G": netG.state_dict(),
                            "state_dict_D": netD.state_dict(),
                            "optimizer_G": optimizerG.state_dict(),
                            "optimizer_D": optimizerD.state_dict()
                            }, "checkpoints_celebA_dcgan/iters_{}.pth".format(iters))

            iters += 1

    save_learning_curves(G_losses, D_losses)

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig("./Output/Real_and_Fake_Images.png")
    plt.show()

    # save models during training
    print("----- Saving models ----")
    torch.save({"iteration": iters,
                "state_dict_G": netG.state_dict(),
                "state_dict_D": netD.state_dict(),
                "optimizer_G": optimizerG.state_dict(),
                "optimizer_D": optimizerD.state_dict()}
               , args.model_path)


if __name__ == "__main__":
    args = get_arguments()
    train_GAN(args)
