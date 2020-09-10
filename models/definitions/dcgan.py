"""DCGAN implementation

    Note1:
        Many implementations out there, including PyTorch's official, did certain deviations from the original arch,
        without clearly explaining why they did it. PyTorch for example uses 512 channels initially instead of 1024.

    Note2:
        Small modification I did compared to the original paper is used kernel size = 4 as I can't get 64x64
        output spatial dimension with 5 no matter the padding setting. I noticed others did the same thing.

"""

from torch import nn
import numpy as np
import utils.utils as utils

from utils.constants import LATENT_SPACE_DIM


def dcgan_block(in_channels, out_channels, normalize=True, activation=None):
    layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU() if activation is None else activation)
    return layers


class GenerativeNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Constants as defined in the DCGAN paper
        num_channels_per_layer = [1024, 512, 256, 128, 3]
        self.init_volume_shape = (num_channels_per_layer[0], 4, 4)

        self.linear = nn.Linear(LATENT_SPACE_DIM, num_channels_per_layer[0] * np.prod(self.init_volume_shape[1:]))

        self.net = nn.Sequential(
            *dcgan_block(num_channels_per_layer[0], num_channels_per_layer[1]),
            *dcgan_block(num_channels_per_layer[1], num_channels_per_layer[2]),
            *dcgan_block(num_channels_per_layer[2], num_channels_per_layer[3]),
            *dcgan_block(num_channels_per_layer[3], num_channels_per_layer[4], normalize=False, activation=nn.Tanh())
        )

    def forward(self, latent_vector_batch):
        # Project from the space with dimensionality 100 into the space with dimensionality 1024 * 4 * 4, linear algebra
        # and reshape into volume
        latent_vector_batch_projected = self.linear(latent_vector_batch)
        latent_vector_batch_projected_reshaped = latent_vector_batch_projected.view(latent_vector_batch_projected.shape[0], *self.init_volume_shape)

        return self.net(latent_vector_batch_projected_reshaped)
