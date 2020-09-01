import os


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_mnist_dataset(dataset_path):
    # It's good to normalize the images to [-1, 1] range https://github.com/soumith/ganhacks
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])
    return datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)


def plot_single_img_from_tensor_batch(batch):
    img = batch[0].numpy()
    img = np.repeat(np.moveaxis(img, 0, 2), 3, axis=2)
    print(img.shape, np.min(img), np.max(img))
    plt.imshow(img)
    plt.show()


# todo: use Goodfellow's original MLP architectures
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
        return x


def prepare_nets(device):
    d_net = DiscriminatorNet().train().to(device)
    g_net = GeneratorNet().train().to(device)
    return d_net, g_net


# todo: try out SGD for discriminator net
def prepare_optimizers(d_net, g_net):
    d_opt = Adam(d_net.parameters())
    g_opt = Adam(g_net.parameters())
    return d_opt, g_opt


if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'MNIST')
    mnist_dataset = get_mnist_dataset(dataset_path)
    mnist_data_loader = DataLoader(mnist_dataset, batch_size=5, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    d_net, g_net = prepare_nets(device)
    d_opt, g_opt = prepare_optimizers(d_net, g_net)
    loss = nn.BCELoss()

    for batch, label in mnist_data_loader:
        plot_single_img_from_tensor_batch(batch)
