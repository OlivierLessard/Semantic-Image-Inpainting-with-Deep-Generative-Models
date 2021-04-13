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
from load_folder import celeba_dataset_dataloader
from dcgan import weights_init
import torchvision.utils as vutils
from dcgan import Discriminator
from visualization import save_learning_curves
import matplotlib.animation as animation
from IPython.display import HTML
from dcgan import Generator, Discriminator


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SVHN")
    parser.add_argument("--mask-type", type=str, default="Center")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-path", type=str, default="./checkpoints/dcgan.pth")
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

    parser.add_argument("--model-path", type=str, default="checkpoints/dcgan.pth")
    parser.add_argument("--z-iteration", type=int, default=1000)

    args = parser.parse_args()
    return args


def z_optimization(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    print("device:", device)

    # dataloader for corrupted images


    # load models
    netG = Generator(args).to(device)
    netD = Discriminator(args).to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    checkpoint = torch.load(args.model_path)
    netG.load_state_dict(checkpoint['state_dict_G'])
    netD.load_state_dict(checkpoint['state_dict_D'])
    optimizerG.load_state_dict(checkpoint['optimizer_G'])
    optimizerD.load_state_dict(checkpoint['optimizer_D'])

    netG.eval()
    netD.eval()

    # freeze G and D

    # get z^(0)
    z = torch.randn(1, args.nz, 1, 1, device=device)
    z.requires_grad = True
    optimizerZ = optim.Adam(z, lr=args.lr, betas=(args.beta1, 0.999))

    print("Starting inpainting ...")
    for i, (corrupted_images, original_images, masks, weighted_masks) in enumerate(dataloader):


        print("Starting backprop to input:  z optimization ...")
        for iter in range(args.z_iteration):
            continue

        # blend corrupted image with G(z)

        # save image

    return None


if __name__ == '__main__':
    args = get_arguments()
    z_optimization(args)


