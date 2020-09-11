"""DCGAN implementation

    Note1:
        Many implementations out there, including PyTorch's official, did certain deviations from the original arch,
        without clearly explaining why they did it. PyTorch for example uses 512 channels initially instead of 1024.

    Note2:
        Small modification I did compared to the original paper is used kernel size = 4 as I can't get 64x64
        output spatial dimension with 5 no matter the padding setting. I noticed others did the same thing.

        Also I'm not doing 0-centered normal weight initialization - it actually gives far worse results.
        Batch normalization, in general, reduced the need for smart initialization but it obviously still matters.

"""

import torch
from torch import nn
import numpy as np


from utils.constants import LATENT_SPACE_DIM


def dcgan_upsample_block(in_channels, out_channels, normalize=True, activation=None):
    # Bias set to True gives unnatural color casts
    layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    # There were debates to whether BatchNorm should go before or after the activation function, in my experiments it
    # did not matter. Goodfellow also had a talk where he mentioned that it should not matter.
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU() if activation is None else activation)
    return layers


class ConvolutionalGenerativeNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Constants as defined in the DCGAN paper
        num_channels_per_layer = [1024, 512, 256, 128, 3]
        self.init_volume_shape = (num_channels_per_layer[0], 4, 4)

        # Both with and without bias gave similar results
        self.linear = nn.Linear(LATENT_SPACE_DIM, num_channels_per_layer[0] * np.prod(self.init_volume_shape[1:]))

        self.net = nn.Sequential(
            *dcgan_upsample_block(num_channels_per_layer[0], num_channels_per_layer[1]),
            *dcgan_upsample_block(num_channels_per_layer[1], num_channels_per_layer[2]),
            *dcgan_upsample_block(num_channels_per_layer[2], num_channels_per_layer[3]),
            *dcgan_upsample_block(num_channels_per_layer[3], num_channels_per_layer[4], normalize=False, activation=nn.Tanh())
        )

    def forward(self, latent_vector_batch):
        # Project from the space with dimensionality 100 into the space with dimensionality 1024 * 4 * 4
        # -> basic linear algebra (huh you thought you'll never need math?) and reshape into a 3D volume
        latent_vector_batch_projected = self.linear(latent_vector_batch)
        latent_vector_batch_projected_reshaped = latent_vector_batch_projected.view(latent_vector_batch_projected.shape[0], *self.init_volume_shape)

        return self.net(latent_vector_batch_projected_reshaped)


def dcgan_downsample_block(in_channels, out_channels, normalize=True, activation=None, padding=1):
    layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=padding, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2) if activation is None else activation)
    return layers


class ConvolutionalDiscriminativeNet(nn.Module):

    def __init__(self):
        super().__init__()

        num_channels_per_layer = [3, 128, 256, 512, 1024, 1]

        # Since the last volume has a shape = 1024x4x4, we can do 1 more block and since it has a 4x4 kernels it will
        # collapse the spatial dimension into 1x1 and putting channel number to 1 and padding to 0 we get a scalar value
        # that we can pass into Sigmoid - effectively simulating a fully connected layer.
        self.net = nn.Sequential(
            *dcgan_downsample_block(num_channels_per_layer[0], num_channels_per_layer[1], normalize=False),
            *dcgan_downsample_block(num_channels_per_layer[1], num_channels_per_layer[2]),
            *dcgan_downsample_block(num_channels_per_layer[2], num_channels_per_layer[3]),
            *dcgan_downsample_block(num_channels_per_layer[3], num_channels_per_layer[4]),
            *dcgan_downsample_block(num_channels_per_layer[4], num_channels_per_layer[5], normalize=False, activation=nn.Sigmoid(), padding=0),
        )

    def forward(self, img_batch):
        return self.net(img_batch)


# Hurts the peformance in all my experiments, leaving it here as a proof that I tried it and it didn't give good results
# Batch normalization in general reduces the need for smart initialization - that's one of it's main advantages.
def weights_init_normal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        # It wouldn't make sense to make this 0-centered normal distribution as it would clamp the outputs to 0
        # that's why it's 1-centered normal distribution with std dev of 0.02 as specified in the paper
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


