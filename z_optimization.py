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
import torch.nn.functional as F
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
    parser.add_argument("--blending-steps", type=int, default=5000)
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
    parser.add_argument("--lr", type=float, default=0.002)

    # parser.add_argument("--z-iteration", type=int, default=1000)
    parser.add_argument("--z-iteration", type=int, default=100)

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
    output = 1 - output / (window_size ** 2 - 1)

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
    output = 1 - (output / (window_size ** 2 - 1))

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
    return corrupted_images, masks


def context_loss(corrupted_images, generated_images, original_weigths, mask_type="central", weighted=True):
    """
    :param corrupted_images: (batch_size, 3, 64, 64)
    :param generated_images: (batch_size, 3, 64, 64)
    :param mask_type:
    :param weighted:
    :return: (batch_size,)
    """
    corrupted_images = corrupted_images.cuda()

    weights = torch.from_numpy(original_weigths)  # ndarray (3, 64, 64)
    weights = torch.unsqueeze(weights, dim=0).cuda()
    weights = torch.repeat_interleave(weights, repeats=corrupted_images.shape[0], dim=0)

    # return torch.sum(torch.abs((generated_images-corrupted_images)*weights), dim=(1, 2, 3))
    return torch.sum(torch.abs((generated_images-corrupted_images)*weights))


def save_images_during_opt(fake_image, corrupted_images, i, iter):
    if not os.path.exists("./Output/z_optimization/"):
        os.mkdir("./Output/z_optimization/")
    first_image = fake_image[0].permute(1, 2, 0).cpu().detach().numpy()
    plt.title("fake image [0] iter 0 ")
    plt.imshow(first_image)
    first_image = (first_image - np.min(first_image))
    first_image = first_image / np.max(first_image)
    plt.imsave("./Output/z_optimization/Batch_{}_fake_iter_{}.jpg".format(i, iter), first_image)
    # plt.show()

    if iter == 0:
        first_image = corrupted_images[0].permute(1, 2, 0).cpu().detach().numpy()
        plt.title("real image [0] iter 0 ")
        plt.imshow(first_image)
        first_image = (first_image - np.min(first_image))
        first_image = first_image / np.max(first_image)
        plt.imsave("./Output/z_optimization/Batch_{}_real_iter_{}.jpg".format(i, iter), first_image)
        # plt.show()
    return None


def image_gradient(image):
    a = torch.Tensor([[[[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]]]]).cuda()
    a = torch.repeat_interleave(a, repeats=3, dim=1)
    b = torch.Tensor([[[[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]]]).cuda()
    b = torch.repeat_interleave(b, repeats=3, dim=1)
    G_x = F.conv2d(image, a, padding=1)
    G_y = F.conv2d(image, b, padding=1)
    return G_x, G_y


def poisson_blending():
    # load last batch
    masks = torch.load('./Output/tmp/masks.pt', map_location=torch.device('cuda:0'))
    original_fake_images = torch.load('./Output/tmp/fake_images.pt', map_location=torch.device('cuda:0'))
    fake_pixels = torch.load('./Output/tmp/fake_images.pt', map_location=torch.device('cuda:0'))
    corrupted_images = torch.load('./Output/tmp/corrupted_images.pt', map_location=torch.device('cuda:0'))

    # define opt for fake_pixels
    initial_guess = masks * corrupted_images + (1-masks) * fake_pixels
    optimizer_blending = optim.Adam([fake_pixels], lr=0.0001)

    # We want the gradient of image_optimum to be like this one
    target_grad_x, target_grad_y = image_gradient(original_fake_images)

    criterion = nn.MSELoss()

    mask_1d = torch.from_numpy(create_weights_one_channel()).cuda()

    print("Starting Poisson blending ...")
    for epoch in range(args.blending_steps):
        optimizer_blending.zero_grad()

        # compute loss and update
        image_optimum = masks * corrupted_images + (1 - masks) * fake_pixels
        optimum_grad_x, optimum_grad_y = image_gradient(image_optimum)

        # blending_loss_x = criterion(target_grad_x*(1-mask_1d), optimum_grad_x*(1-mask_1d))
        # blending_loss_y = criterion(target_grad_y*(1-mask_1d), optimum_grad_y*(1-mask_1d))
        blending_loss_x = criterion(target_grad_x, optimum_grad_x)
        blending_loss_y = criterion(target_grad_y, optimum_grad_y)

        # add the gradients
        blending_loss = blending_loss_x + blending_loss_y
        blending_loss.backward(retain_graph=True)  # retain_graph=True
        # update image_optimum
        optimizer_blending.step()

        print("[Epoch: {}/{}] \t[Blending loss: {:.3f}]   \r".format(1+epoch, args.blending_steps, blending_loss), end="")

    # bring back the original pixels
    blend_image = masks * corrupted_images + (1 - masks) * image_optimum
    plt.title("blend_image[0]")
    plt.imshow(blend_image[0].permute(1, 2, 0).cpu().detach().numpy())
    plt.show()
    plt.title("corrupted_images[0]")
    plt.imshow(corrupted_images[0].permute(1, 2, 0).cpu().detach().numpy())
    plt.show()
    return initial_guess, blend_image


def save_blend_images(corrupted_images, initial_guess, blend_images, save_count):
    if not os.path.exists("./Output/Blend/"):
        os.mkdir("./Output/Blend/")
    for i in range(blend_images.shape[0]):
        image = corrupted_images[i].permute(1, 2, 0).cpu().detach().numpy()
        plt.title("corrupted_images {}".format(i+save_count))
        plt.imshow(image)
        image = (image - np.min(image))
        image = image / np.max(image)
        plt.imsave("./Output/Blend/Image_{}_corrupted.jpg".format(i+save_count), image)
        #plt.show()

        image = initial_guess[i].permute(1, 2, 0).cpu().detach().numpy()
        plt.title("initial_guess {}".format(i+save_count))
        plt.imshow(image)
        image = (image - np.min(image))
        image = image / np.max(image)
        plt.imsave("./Output/Blend/Image_{}_initial_guess.jpg".format(i+save_count), image)
        # plt.show()

        image = blend_images[i].permute(1, 2, 0).cpu().detach().numpy()
        plt.title("blended image {}".format(i+save_count))
        plt.imshow(image)
        image = (image - np.min(image))
        image = image / np.max(image)
        plt.imsave("./Output/Blend/Image_{}_blend.jpg".format(i+save_count), image)
        #plt.show()


def save_tensors(masks, fake_image, corrupted_images):
    if not os.path.exists("./Output/tmp/"):
        os.mkdir("./Output/tmp/")
    torch.save(masks, './Output/tmp/masks.pt')
    torch.save(fake_image, './Output/tmp/fake_images.pt')
    torch.save(corrupted_images, './Output/tmp/corrupted_images.pt')
    return None


def z_optimization(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    print("device:", device)

    # dataloader for original images
    dataset, dataloader = celeba_dataset_dataloader(args)

    # load models
    netG = Generator(args).to(device)
    netD = Discriminator(args).to(device)
    checkpoint = torch.load(args.model_path)
    netG.load_state_dict(checkpoint['state_dict_G'])
    netD.load_state_dict(checkpoint['state_dict_D'])

    netG.eval()
    netD.eval()

    # Initialize BCELoss function
    criterion = nn.BCELoss(reduction='sum')

    # freeze G and D
    print("Starting inpainting ...")
    save_count = 0
    nb_batch_to_inpaint = 1
    for i, data_and_labels in enumerate(dataloader):
        if i == nb_batch_to_inpaint:
            break

        original_images = data_and_labels[0]
        corrupted_images, masks = apply_mask(original_images, mask_type="central")
        original_weigths = create_weights_three_channel()

        # get z^(0)
        b_size = original_images.size(0)
        z = torch.randn(b_size, args.nz, 1, 1, device=device)
        z.requires_grad = True
        optimizerZ = optim.Adam([z], lr=args.lr, betas=(args.beta1, 0.999))

        print("Starting backprop to input:  z optimization ...")
        for iter in range(args.z_iteration):
            optimizerZ.zero_grad()

            fake_image = netG(z)
            # check current fake_image
            plots = False
            if plots:
                plt.title("corrupted_images[0]")
                plt.imshow(corrupted_images[0].permute(1, 2, 0))
                # plt.show()
                plt.title("fake image [0]")
                plt.imshow(fake_image[0].permute(1, 2, 0).cpu().detach().numpy())
                # plt.show()
            if iter == 0 or iter == args.z_iteration-1:
                save_images_during_opt(fake_image, corrupted_images, i, iter)

            # compute losses
            c_loss = context_loss(corrupted_images, fake_image, original_weigths, mask_type="central", weighted=True)
            fake_prediction = netD(fake_image)
            fake_prediction = torch.squeeze(fake_prediction)
            real_label = torch.ones(corrupted_images.shape[0], device=device)
            prior_loss = criterion(fake_prediction, real_label)
            total_loss = c_loss + args.prior_weight*prior_loss

            # update z
            total_loss.backward()
            optimizerZ.step()

            print("iters {}, loss {}".format(iter, total_loss))

        # blend corrupted image with G(z)
        save_tensors(masks, fake_image, corrupted_images)
        initial_guess, blend_images = poisson_blending()

        # save images
        save_blend_images(corrupted_images, initial_guess, blend_images, save_count)
        save_count += b_size

    return None


if __name__ == '__main__':
    args = get_arguments()
    z_optimization(args)


