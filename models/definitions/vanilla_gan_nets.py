import torch
from torch import nn

# todo: add latent dim
# from .contants import LA


MNIST_IMG_SIZE = 28


def get_vanilla_nets(device):
    d_net = DiscriminatorNet().train().to(device)
    g_net = GeneratorNet().train().to(device)
    return d_net, g_net


class DiscriminatorNet(torch.nn.Module):
    """
    4-layer MLP discriminative neural network
    """

    def __init__(self):
        super().__init__()
        num_neurons_per_layer = [784, 1024, 512, 256, 1]
        leaky_relu_coeff = 0.2
        dropout_prob = 0.3

        self.layer1 = nn.Sequential(
            nn.Linear(num_neurons_per_layer[0], num_neurons_per_layer[1]),
            nn.LeakyReLU(leaky_relu_coeff),
            nn.Dropout(dropout_prob)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_neurons_per_layer[1], num_neurons_per_layer[2]),
            nn.LeakyReLU(leaky_relu_coeff),
            nn.Dropout(dropout_prob)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(num_neurons_per_layer[2], num_neurons_per_layer[3]),
            nn.LeakyReLU(leaky_relu_coeff),
            nn.Dropout(dropout_prob)
        )
        self.layer4 = nn.Sequential(
            torch.nn.Linear(num_neurons_per_layer[3], num_neurons_per_layer[4]),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class GeneratorNet(torch.nn.Module):
    """
    Simple 4-layer MLP generative neural network
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        num_neurons_per_layer = [100, 256, 512, 1024, 784]
        leaky_relu_coeff = 0.2

        self.layer1 = nn.Sequential(
            nn.Linear(num_neurons_per_layer[0], num_neurons_per_layer[1]),
            nn.LeakyReLU(leaky_relu_coeff)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_neurons_per_layer[1], num_neurons_per_layer[2]),
            nn.LeakyReLU(leaky_relu_coeff)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(num_neurons_per_layer[2], num_neurons_per_layer[3]),
            nn.LeakyReLU(leaky_relu_coeff)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(num_neurons_per_layer[3], num_neurons_per_layer[4]),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.view(x.shape[0], 1, MNIST_IMG_SIZE, MNIST_IMG_SIZE)