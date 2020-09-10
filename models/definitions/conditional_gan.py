"""Conditional GAN (cGAN) implementation.

    It's completely the same architecture as vanilla GAN just with additional conditioning vector on the input.

    Note: I could have merged this file with vanilla_gan.py and made the conditioning vector be an optional input,
    but I decided not to for ease of understanding for the beginners. Otherwise it could get a bit confusing.
"""


import torch
from torch import nn


from utils.constants import LATENT_SPACE_DIM, MNIST_IMG_SIZE, MNIST_NUM_CLASSES
from .vanilla_gan import vanilla_block


class ConditionalGeneratorNet(torch.nn.Module):
    """Simple 4-layer MLP generative neural network.

    By default it works for MNIST size images (28x28).

    There are many ways you can construct generator to work on MNIST.
    Even without normalization layers it will work ok. Even with 5 layers it will work ok, etc.

    It's generally an open-research question on how to evaluate GANs i.e. quantify that "ok" statement.

    People tried to automate the task using IS (inception score, often used incorrectly), etc.
    but so far it always ends up with some form of visual inspection (human in the loop).

    """

    def __init__(self, img_shape=(MNIST_IMG_SIZE, MNIST_IMG_SIZE)):
        super().__init__()
        self.generated_img_shape = img_shape
        # We're adding the conditioning vector (hence +MNIST_NUM_CLASSES) which will directly control
        # which MNIST class we should generate. We did not have this control in the original (vanilla) GAN.
        # If that vector = [1., 0., ..., 0.] we generate 0, if [0., 1., 0., ..., 0.] we generate 1, etc.
        num_neurons_per_layer = [LATENT_SPACE_DIM + MNIST_NUM_CLASSES, 256, 512, 1024, img_shape[0] * img_shape[1]]

        self.net = nn.Sequential(
            *vanilla_block(num_neurons_per_layer[0], num_neurons_per_layer[1]),
            *vanilla_block(num_neurons_per_layer[1], num_neurons_per_layer[2]),
            *vanilla_block(num_neurons_per_layer[2], num_neurons_per_layer[3]),
            *vanilla_block(num_neurons_per_layer[3], num_neurons_per_layer[4], normalize=False, activation=nn.Tanh())
        )

    def forward(self, latent_vector_batch, one_hot_conditioning_vector_batch):
        img_batch_flattened = self.net(torch.cat((latent_vector_batch, one_hot_conditioning_vector_batch), 1))
        # just un-flatten using view into (N, 1, 28, 28) shape for MNIST
        return img_batch_flattened.view(img_batch_flattened.shape[0], 1, *self.generated_img_shape)


class ConditionalDiscriminatorNet(torch.nn.Module):
    """Simple 3-layer MLP discriminative neural network. It should output probability 1. for real images and 0. for fakes.

    By default it works for MNIST size images (28x28).

    Again there are many ways you can construct discriminator network that would work on MNIST.
    You could use more or less layers, etc. Using normalization as in the DCGAN paper doesn't work well though.

    """

    def __init__(self, img_shape=(MNIST_IMG_SIZE, MNIST_IMG_SIZE)):
        super().__init__()
        # Same as above using + MNIST_NUM_CLASSES we add support for the conditioning vector
        num_neurons_per_layer = [img_shape[0] * img_shape[1] + MNIST_NUM_CLASSES, 512, 256, 1]

        self.net = nn.Sequential(
            *vanilla_block(num_neurons_per_layer[0], num_neurons_per_layer[1], normalize=False),
            *vanilla_block(num_neurons_per_layer[1], num_neurons_per_layer[2], normalize=False),
            *vanilla_block(num_neurons_per_layer[2], num_neurons_per_layer[3], normalize=False, activation=nn.Sigmoid())
        )

    def forward(self, img_batch, one_hot_conditioning_vector_batch):
        img_batch_flattened = img_batch.view(img_batch.shape[0], -1)  # flatten from (N,1,H,W) into (N, HxW)
        # One hot conditioning vector batch is of shape (N, 10) for MNIST
        conditioned_input = torch.cat((img_batch_flattened, one_hot_conditioning_vector_batch), 1)
        return self.net(conditioned_input)



