import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset
from torchvision import transforms
import glob, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from itertools import chain
from torchvision import utils
import argparse
from load_folder import celeba_dataset_dataloader

SAVE_PER_TIMES = 100


class Generator(torch.nn.Module):
    def __init__(self, channels = 3):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
                        # input is Z, going into a convolution
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1)
            )
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels = 3):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class WGAN_GP(object):
    def __init__(self):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator()
        self.D = Discriminator()
        self.C = 3

        # Check if cuda is available

        self.check_cuda(True)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 128

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.number_of_images = 10

        self.generator_iters = 1
        self.critic_iter = 5
        self.lambda_term = 10

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train(self, train_loader):

        print("starting")

        self.t_begin = t.time()
        self.file = open("inception_score_graph.txt", "w")

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        i = 0

        for images, labels in train_loader:
            for g_iter in range(self.generator_iters):
                # Requires grad, Generator requires_grad = False
                for p in self.D.parameters():
                    p.requires_grad = True

                d_loss_real = 0
                d_loss_fake = 0
                Wasserstein_D = 0
                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
                for d_iter in range(self.critic_iter):
                    self.D.zero_grad()

                    images = self.data.__next__()
                    # Check for batch to have full batch_size
                    if (images.size()[0] != self.batch_size):
                        continue

                    z = torch.rand((self.batch_size, 100, 1, 1))

                    images, z = self.get_torch_variable(images), self.get_torch_variable(z)

                    # Train discriminator
                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    d_loss_real = self.D(images)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    # Train with fake images
                    z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))

                    fake_images = self.G(z)
                    d_loss_fake = self.D(fake_images)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                    gradient_penalty.backward()


                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_D = d_loss_real - d_loss_fake
                    self.d_optimizer.step()
                    print('  Discriminator iteration: {}/{}, loss_fake: {}, loss_real: {}'.format(d_iter, self.critic_iter, d_loss_fake, d_loss_real))

                # Generator update
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                # train generator
                # compute loss with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_images = self.G(z)
                g_loss = self.D(fake_images)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                g_cost = -g_loss
                self.g_optimizer.step()
                print('Generator iteration: {}/{}, g_loss: {}'.format(g_iter, self.generator_iters, g_loss))

            if i % 100 == 0:
              self.save_model()
            i += 1

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))


    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), '/content/drive/MyDrive/checkpoints_celebA_dcgan/generator.pth')
        torch.save(self.D.state_dict(), '/content/drive/MyDrive/checkpoints_celebA_dcgan/discriminator.pth')
        print('Models save to ./generator.pth & ./discriminator.pth ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")

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

    # parser.add_argument("--blending-steps", type=int, default=5000)
    parser.add_argument("--blending-steps", type=int, default=500)
    # parser.add_argument("--z-iteration", type=int, default=1000)
    parser.add_argument("--z-iteration", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    # get dataloader
    dataset, dataloader = celeba_dataset_dataloader(args)

    # get model
    wgan = WGAN_GP()

    # train
    wgan.train(dataloader)

    # load the model after training
    # netG = Generator()#.to(device)
    # netD = Discriminator()#.to(device)
    # checkpointG = torch.load("/content/drive/MyDrive/checkpoints_celebA_dcgan/generator.pth")
    # checkpointG = torch.load("/content/drive/MyDrive/checkpoints_celebA_dcgan/discriminator.pth")
    # netG.load_state_dict(checkpointG['state_dict'])
    # netD.load_state_dict(checkpointD['state_dict'])
