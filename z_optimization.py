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

    parser.add_argument("--z-iteration", type=int, default=1000)

    args = parser.parse_args()
    return args


def create_weights_one_channel():
    mask = np.ones((64, 64), dtype=np.float32)

    x = 25
    y = 25
    h = 40
    w = 40

    mask[x:h, y:w] = 0

    window_size = 7

    max_shift = window_size // 2
    output = np.zeros_like(mask)

    print("calculating weights")
    for i in range(-max_shift, max_shift + 1):
        for j in range(-max_shift, max_shift + 1):
            if i != 0 or j != 0:
                output += np.roll(mask, (i, j), axis=(0, 1))
    output = 1 - output / (window_size * 2 - 1)

    final = output*mask
    return final


def create_weights_three_channel():
    mask = np.ones((3, 64, 64), dtype=np.float32)

    x = 25
    y = 25
    h = 40
    w = 40

    mask[:, x:h, y:w] = 0

    window_size = 7

    max_shift = window_size // 2
    output = np.zeros_like(mask)

    print("calculating weights")
    for i in range(-max_shift, max_shift + 1):
        for j in range(-max_shift, max_shift + 1):
            if i != 0 or j != 0:
                output += np.roll(mask, (i, j), axis=(1, 2))
    output = 1 - output / (window_size * 2 - 1)

    final = output*mask
    return final


def apply_mask(original_images, mask_type="central"):
    if mask_type == "central":
        width = original_images.shape[2]
        height = original_images.shape[3]
        mask_position_x = int(width/2)
        mask_position_y = int(height/2)
        mask_width = 15
        masks = torch.ones_like(original_images, dtype=torch.float32)
        masks.cuda()
        masks[:, :, int(mask_position_x - mask_width/2):int(mask_position_x + mask_width/2), int(mask_position_y - mask_width/2):int(mask_position_y + mask_width/2)] = 0
        corrupted_images = original_images * masks

        # plt.imshow(corrupted_images[0].permute(1, 2, 0))
        # plt.show()
    return corrupted_images


def context_loss(corrupted_images, generated_images, mask_type="central", weighted=True):
    corrupted_images = corrupted_images.cuda()

    weights = torch.from_numpy(create_weights_three_channel())
    weights = torch.unsqueeze(weights, dim=0).cuda()
    weights = torch.repeat_interleave(weights, repeats=corrupted_images.shape[0], dim=0)
    return (generated_images-corrupted_images)*weights


def z_optimization(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    print("device:", device)

    # dataloader for original images
    dataset, dataloader = celeba_dataset_dataloader(args)

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
    print("Starting inpainting ...")
    for i, data_and_labels in enumerate(dataloader):
        original_images = data_and_labels[0]
        corrupted_images = apply_mask(original_images, mask_type="central")

        # get z^(0)
        b_size = original_images.size(0)
        z = torch.randn(b_size, args.nz, 1, 1, device=device)
        z.requires_grad = True
        optimizerZ = optim.Adam([z], lr=args.lr, betas=(args.beta1, 0.999))

        print("Starting backprop to input:  z optimization ...")
        for iter in range(args.z_iteration):
            optimizerZ.zero_grad()

            fake_image = netG(z)
            #plt.imshow(corrupted_images[0].permute(1, 2, 0))
            c_loss = context_loss(corrupted_images, fake_image, mask_type="central", weighted=True)
            prior_loss = 0
            total_loss = c_loss + args.prior_weight*prior_loss
            continue

        # blend corrupted image with G(z)

        # save image

    return None


if __name__ == '__main__':
    args = get_arguments()
    z_optimization(args)


