import argparse
from data import svhn_dataset_dataloader
from torch import nn, optim
import torch
from matplotlib import pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from data import celeba_dataset_dataloader
from dcgan import Generator, Discriminator
import training_wgan_gp


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="svhn")    # svhn, CelebA
    parser.add_argument("--mask-type", type=str, default="center")  # center, random, pattern, half
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--wgan", type=bool, default=False)
    parser.add_argument("--train-data-dir", type=str, default="./Datasets/CelebA/")

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--optim-steps", type=int, default=1500)
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

    parser.add_argument("--blending-steps", type=int, default=5000)
    # parser.add_argument("--blending-steps", type=int, default=500)
    parser.add_argument("--z-iteration", type=int, default=1000)
    # parser.add_argument("--z-iteration", type=int, default=10)

    args = parser.parse_args()
    return args


def create_weights_three_channel(masks, batch_size, mask_type="center"):
    """
    Create the weights (batchsize, channnels, w, h) to apply later for the loss
    :param masks:
    :param batch_size:
    :param mask_type:
    :return:
    """
    if mask_type == "center":
        x = 20
        y = 20
        h = 45
        w = 45

        mask = np.ones((3, 64, 64), dtype=np.float32)
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

        final = output * mask

        # create batch weights (128, 3, 64, 64)
        weights = torch.from_numpy(final)  # ndarray (3, 64, 64)
        weights = torch.unsqueeze(weights, dim=0).cuda()
        weights = torch.repeat_interleave(weights, repeats=batch_size, dim=0)

    elif mask_type == "random" or mask_type == "pattern" or mask_type == "half":
        weights = torch.zeros_like(masks)  # (128, 3, 64, 64)

        print("calculating weights")
        window_size = 7
        max_shift = window_size // 2

        for b in range(masks.shape[0]):

            mask = masks[b].squeeze().cpu().detach().numpy()
            output = np.zeros_like(mask)
            for i in range(-max_shift, max_shift + 1):
                for j in range(-max_shift, max_shift + 1):
                    if i != 0 or j != 0:
                        output += np.roll(mask, (i, j), axis=(1, 2))
            output = 1 - (output / (window_size ** 2 - 1))
            final = output * mask

            weights[b] = torch.from_numpy(final)

    return weights  # (128, 3, 64, 64)


def apply_mask(original_images, mask_type="center"):
    """
    :param original_images: values in range [-1, 1]!!
    :param mask_type: {0, 1}
    :return: masked pixels are set to -1 for all channels !!
    """
    if mask_type == "center":
        width = original_images.shape[2]
        height = original_images.shape[3]
        mask_position_x = int(width/2)
        mask_position_y = int(height/2)
        mask_width = 25
        masks = torch.ones_like(original_images, dtype=torch.float32)
        masks.cuda()
        masks[:, :, int(mask_position_x - mask_width/2):int(mask_position_x + mask_width/2), int(mask_position_y - mask_width/2):int(mask_position_y + mask_width/2)] = 0

        # set masked pixels to -1
        black_pixels = -1*torch.ones_like(original_images)
        corrupted_images = torch.where(masks != 0, original_images, black_pixels)

    elif mask_type == "random":  # 80% missing, random mask
        masks = torch.FloatTensor(original_images.shape).uniform_() > 0.8
        masks[:, 1, :, :] = masks[:, 0, :, :]
        masks[:, 2, :, :] = masks[:, 0, :, :]
        masks = masks.float().cuda()
        original_images = original_images.cuda()

        # set masked pixels to -1
        black_pixels = -1*torch.ones_like(original_images)
        corrupted_images = torch.where(masks != 0, original_images, black_pixels)

    elif mask_type == "pattern":  # 80% missing, random mask
        masks = torch.FloatTensor(original_images.shape).uniform_() > 0.2
        masks[:, 1, :, :] = masks[:, 0, :, :]
        masks[:, 2, :, :] = masks[:, 0, :, :]
        masks = masks.float().cuda()
        original_images = original_images.cuda()

        # set masked pixels to -1
        black_pixels = -1*torch.ones_like(original_images)
        corrupted_images = torch.where(masks != 0, original_images, black_pixels)

    elif mask_type == "half":  # 80% missing, random mask
        masks = torch.ones_like(original_images)
        masks[:, :, 0:32, 0:64] = 0
        masks = masks.float().cuda()
        original_images = original_images.cuda()

        # set masked pixels to -1
        black_pixels = -1*torch.ones_like(original_images)
        corrupted_images = torch.where(masks != 0, original_images, black_pixels)

    return corrupted_images, masks


def context_loss(corrupted_images, generated_images, weights):
    """
    :param corrupted_images: (batch_size, 3, 64, 64)
    :param generated_images: (batch_size, 3, 64, 64)
    :param weigths:          (batch_size, 3, 64, 64)
    :param mask_type: str
    :return: (batch_size,)
    """
    corrupted_images = corrupted_images.cuda()

    return torch.sum(torch.abs((generated_images-corrupted_images)*weights))


def save_images_during_opt(args, fake_image, corrupted_images, i, iter):
    if not args.wgan:
        z_opt_path = os.path.join("./Output_{}_dcgan".format(args.dataset), "z_optimization")
    else:
        if not os.path.exists("./Output_{}_wgan/".format(args.dataset)):
            os.mkdir("./Output_{}_wgan/".format(args.dataset))
        z_opt_path = os.path.join("./Output_{}_wgan/".format(args.dataset), "z_optimization")
    if not os.path.exists(z_opt_path):
        os.mkdir(z_opt_path)
    z_opt_mask_path = os.path.join(z_opt_path, args.mask_type)
    if not os.path.exists(z_opt_mask_path):
        os.mkdir(z_opt_mask_path)

    # save current fake image
    first_image = fake_image[0].permute(1, 2, 0).cpu().detach().numpy()
    plt.title("fake image [0] iter 0 ")
    normalize_image = (first_image - np.min(first_image))/(np.max(first_image) - np.min(first_image))
    plt.imshow(normalize_image)
    save_path = os.path.join(z_opt_mask_path, "Batch_{}_fake_iter_{}.jpg".format(i, iter))
    plt.imsave(save_path, normalize_image)
    # plt.show()

    # save the real image only at beginning, won't change
    if iter == 0:
        first_image = corrupted_images[0].permute(1, 2, 0).cpu().detach().numpy()
        plt.title("real image [0] iter 0 ")
        normalize_image = (first_image - np.min(first_image)) / (np.max(first_image) - np.min(first_image))
        plt.imshow(normalize_image)
        save_path = os.path.join(z_opt_mask_path, "Batch_{}_real_iter_{}.jpg".format(i, iter))
        plt.imsave(save_path, normalize_image)
        # plt.show()
    return None


def image_gradient(image):
    lap = torch.Tensor([[[[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]]]]).cuda()
    lap = torch.repeat_interleave(lap, repeats=3, dim=1)
    G = F.conv2d(image, lap, padding=1)
    return G


def poisson_blending():
    """
    Backprop to the fake_pixels to obtain a keep the same gradient in the image as the fake image
    :return: initial and final guess
    """
    # load last batch, pixels in range [-1, 1]
    masks = torch.load('./tmp/masks.pt', map_location=torch.device('cuda:0'))
    original_fake_images = torch.load('./tmp/fake_images.pt', map_location=torch.device('cuda:0'))
    fake_pixels = torch.load('./tmp/fake_images.pt', map_location=torch.device('cuda:0'))
    corrupted_images = torch.load('./tmp/corrupted_images.pt', map_location=torch.device('cuda:0'))

    # define opt for fake_pixels
    initial_guess = torch.where(masks != 0, corrupted_images, fake_pixels)
    optimizer_blending = optim.Adam([fake_pixels], lr=0.0001)

    # We want the gradient of current_blending to be like this one
    target_grad = image_gradient(original_fake_images)

    criterion = nn.MSELoss()

    print("Starting Poisson blending ...")
    for epoch in range(args.blending_steps):
        optimizer_blending.zero_grad()

        # compute blending and gradient
        current_blending = torch.where(masks != 0, corrupted_images, fake_pixels)
        current_grad = image_gradient(current_blending)

        # compute loss
        blending_loss = criterion(target_grad, current_grad)**2

        # update fake_pixels
        blending_loss.backward(retain_graph=True)  # retain_graph=True
        optimizer_blending.step()

        print("[Epoch: {}/{}] \t[Blending loss: {:.3f}]   \r".format(1+epoch, args.blending_steps, blending_loss), end="")

    # bring back the original pixels, sanity check
    blend_image = torch.where(masks != 0, corrupted_images, current_blending)
    # plt.title("blend_image[0]")
    # plt.imshow(blend_image[0].permute(1, 2, 0).cpu().detach().numpy())
    # plt.show()
    # plt.title("corrupted_images[0]")
    # plt.imshow(corrupted_images[0].permute(1, 2, 0).cpu().detach().numpy())
    # plt.show()
    return initial_guess, blend_image


def save_blend_images(args, original_images, corrupted_images, initial_guess, blend_images, save_count):
    if not args.wgan:
        blend_path = os.path.join("./Output_{}_dcgan/".format(args.dataset), "Blend/")
    else:
        blend_path = os.path.join("./Output_{}_wgan/".format(args.dataset), "Blend/")
    if not os.path.exists(blend_path):
        os.mkdir(blend_path)
    blend_mask_path = os.path.join(blend_path, args.mask_type)
    if not os.path.exists(blend_mask_path):
        os.mkdir(blend_mask_path)

    titles = ["original", "corrupted", "initial_guess", "blend"]
    for i in range(blend_images.shape[0]):
        print("Saving images #{}".format(i+save_count))

        image = original_images[i].permute(1, 2, 0).cpu().detach().numpy()
        original_i = image
        plt.title("original_images {}".format(i + save_count))
        image = (image - np.min(image))/(np.max(image) - np.min(image))
        plt.imshow(image)
        save_path = os.path.join(blend_mask_path, "Image_{}_original.png".format(i + save_count))
        plt.imsave(save_path, image, format='png')
        # plt.show()

        image = corrupted_images[i].permute(1, 2, 0).cpu().detach().numpy()
        corrupted_i = image
        plt.title("corrupted_images {}".format(i+save_count))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        plt.imshow(image)
        save_path = os.path.join(blend_mask_path, "Image_{}_corrupted.png".format(i + save_count))
        image = image * (corrupted_i != 0)
        plt.imsave(save_path, image, format='png')
        #plt.show()

        image = initial_guess[i].permute(1, 2, 0).cpu().detach().numpy()
        first_guess_i = image
        plt.title("initial_guess {}".format(i+save_count))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        plt.imshow(image)
        save_path = os.path.join(blend_mask_path, "Image_{}_initial_guess.png".format(i + save_count))
        plt.imsave(save_path, image, format='png')
        # plt.show()

        image = blend_images[i].permute(1, 2, 0).cpu().detach().numpy()
        blend_image_i = image
        plt.title("blended image {}".format(i+save_count))
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        plt.imshow(image)
        save_path = os.path.join(blend_mask_path, "Image_{}_blend.png".format(i + save_count))
        plt.imsave(save_path, image, format='png')
        #plt.show()

        from visualization import show_images
        arrays = [original_i, corrupted_i, first_guess_i, blend_image_i]
        save_name = os.path.join(blend_mask_path, "Image_{}_all.jpg".format(i + save_count))
        show_images(arrays, save_name, cols=1, titles=titles)


def save_tensors(masks, fake_image, corrupted_images):
    if not os.path.exists("./tmp/"):
        os.mkdir("./tmp/")
    torch.save(masks, './tmp/masks.pt')
    torch.save(fake_image, './tmp/fake_images.pt')
    torch.save(corrupted_images, './tmp/corrupted_images.pt')
    return None


def z_optimization(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    print("device:", device)

    # dataloader for original images
    if args.dataset == "CelebA":
        dataset, dataloader = celeba_dataset_dataloader(args)
    elif args.dataset == "svhn":
        dataset, dataloader = svhn_dataset_dataloader(args, split='test')
        # dataset, dataloader = svhn_dataset_dataloader(args, split='train')

    # load models dcgan or wgan
    if not args.wgan:
        netG = Generator(args).to(device)
        netD = Discriminator(args).to(device)
        if args.dataset == "CelebA":
            checkpoint = torch.load("./checkpoints_celebA_dcgan/dcgan.pth")
        if args.dataset == "svhn":
            checkpoint = torch.load("./checkpoints_svhn_dcgan/model.pth")
        netG.load_state_dict(checkpoint['state_dict_G'])
        netD.load_state_dict(checkpoint['state_dict_D'])
    else:
        if args.dataset == "svhn":
            opt = training_wgan_gp.get_args()
            netG = training_wgan_gp.Generator()
            checkpoint = torch.load("./checkpoints_svhn_wgan/wgan_gp.pth")
            netG.load_state_dict(checkpoint["state_dict_G"])
            netG.cuda()

            # test with dcgan D because it's a classifier
            netD = Discriminator(args).to(device)
            checkpoint = torch.load("./checkpoints_svhn_dcgan/model.pth")
            netD.load_state_dict(checkpoint['state_dict_D'])
        else:
            raise NameError('WGAN was not trained with celebA')

    # freeze G and D
    netG.eval()
    netD.eval()

    # Initialize BCELoss function
    criterion = nn.BCELoss(reduction='sum')

    print("Start inpainting ...")
    save_count = 0
    nb_batch_to_inpaint = 2
    for i, data_and_labels in enumerate(dataloader):
        if i == nb_batch_to_inpaint:
            break

        original_images = data_and_labels[0]        # images between [-1, 1]
        corrupted_images, masks = apply_mask(original_images, mask_type=args.mask_type)
        original_weigths = create_weights_three_channel(masks, batch_size=original_images.shape[0], mask_type=args.mask_type)

        # get z^(0)
        current_batch_size = original_images.size(0)
        z = torch.randn(current_batch_size, args.nz, 1, 1, device=device)
        z.requires_grad = True
        optimizerZ = optim.Adam([z], lr=args.lr, betas=(args.beta1, 0.999))

        print("Starting backprop to input:  z optimization ...")
        for iter in range(args.z_iteration):
            optimizerZ.zero_grad()

            fake_image = netG(z.squeeze()) if args.wgan else netG(z)

            if iter == 0 or iter == args.z_iteration-1:
                save_images_during_opt(args, fake_image, corrupted_images, i, iter)

            # compute losses
            c_loss = context_loss(corrupted_images, fake_image, original_weigths)
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
        save_blend_images(args, original_images, corrupted_images, initial_guess, blend_images, save_count)
        save_count += current_batch_size

    return None


if __name__ == '__main__':
    args = get_arguments()

    # set this argument to True if you want to use the WGAN model
    args.wgan = True
    args.dataset = "svhn"

    # inference with all the mask types
    for mask_type in ["center", "pattern", "half", "random"]:
        args.mask_type = mask_type
        z_optimization(args)
