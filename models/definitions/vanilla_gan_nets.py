import torch
from torch import nn

# todo: add latent dim
# from utils.contants import LA


MNIST_IMG_SIZE = 28


def vanilla_block(in_feat, out_feat, normalize=True, activation=None):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU(0.2) if activation is None else activation)
    return layers


class GeneratorNet(torch.nn.Module):
    """
    Simple 4-layer MLP generative neural network.

    There are many ways you can construct generator to work on MNIST.
    Even without normalization layers it will work ok. Even with 5 layers it will work ok, etc.

    It's generally an open-research question on how to evaluate GANs i.e. quantify that "ok" statement.

    People tried to automate the task using IS (inception score, often used incorrectly), etc.
    but so far it always ends up with some form of visual inspection (human in the loop).
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        num_neurons_per_layer = [100, 256, 512, 1024, 784]

        self.net = nn.Sequential(
            *vanilla_block(num_neurons_per_layer[0], num_neurons_per_layer[1]),
            *vanilla_block(num_neurons_per_layer[1], num_neurons_per_layer[2]),
            *vanilla_block(num_neurons_per_layer[2], num_neurons_per_layer[3]),
            *vanilla_block(num_neurons_per_layer[3], num_neurons_per_layer[4], normalize=False, activation=nn.Tanh())
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.shape[0], 1, MNIST_IMG_SIZE, MNIST_IMG_SIZE)


class DiscriminatorNet(torch.nn.Module):
    """
    Simple 3-layer MLP discriminative neural network.

    Again there are many ways you can construct discriminator network that would work on MNIST.
    You could use more or less layers, etc. Using normalization as in DCGAN paper doesn't work well though.
    """

    def __init__(self):
        super().__init__()
        # todo: make this 784 = mnist_size x mnist_size or more generic
        num_neurons_per_layer = [784, 512, 256, 1]

        self.net = nn.Sequential(
            *vanilla_block(num_neurons_per_layer[0], num_neurons_per_layer[1], normalize=False),
            *vanilla_block(num_neurons_per_layer[1], num_neurons_per_layer[2], normalize=False),
            *vanilla_block(num_neurons_per_layer[2], num_neurons_per_layer[3], normalize=False, activation=nn.Sigmoid()),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # flatten from (N,1,H,W) into (N, HxW)
        return self.net(x)



