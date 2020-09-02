import os
import copy


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
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


def same_weights(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def compose_imgs(batch):
    imgs = []
    for img in batch:
        img = np.moveaxis(img.to('cpu').numpy(), 0, 2)
        print(np.min(img), np.max(img), len(np.unique(img)), np.unique(img)[:4])
        img += 1.
        img /= 2.
        imgs.append(np.uint8(img * 255))
    return np.hstack(imgs)


if __name__ == "__main__":
    batch_size = 100
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'MNIST')
    debug_path = os.path.join(os.path.dirname(__file__), 'data', 'debug_dir')
    os.makedirs(debug_path, exist_ok=True)
    mnist_dataset = get_mnist_dataset(dataset_path)
    mnist_data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    d_net, g_net = prepare_nets(device)
    d_opt, g_opt = prepare_optimizers(d_net, g_net)
    loss_fn = nn.BCELoss()
    num_epochs = 200

    ref_noise_batch = torch.randn((5, 100), device=device)
    g_steps = 1
    d_steps = 5

    real_losses = []
    fake_losses = []
    losses = []
    for epoch in range(num_epochs):
        for cnt, (real_batch, label) in enumerate(mnist_data_loader):
            if cnt % 100 == 0:
                print(f'Training. Epoch = {epoch} batch = {cnt}.')

            real_batch = real_batch.view(real_batch.shape[0], -1)
            real_batch = real_batch.to(device)

            # Train discriminator net
            real_predictions = d_net(real_batch)
            real_gt = torch.ones((batch_size, 1), device=device)
            real_loss = loss_fn(real_predictions, real_gt)
            real_loss.backward()
            if cnt % d_steps == 0:
                real_losses.append(real_loss.item())

            noise_batch = torch.randn((batch_size, 100), device=device)
            fake_batch = g_net(noise_batch)
            fake_predictions = d_net(fake_batch.detach())
            fake_gt = torch.zeros((batch_size, 1), device=device)
            fake_loss = loss_fn(fake_predictions, fake_gt)
            fake_loss.backward()
            if cnt % d_steps == 0:
                fake_losses.append(fake_loss.item())

            d_opt.step()
            d_net.zero_grad()

            if cnt % d_steps == 0:
                # Train generator net
                noise_batch = torch.randn((batch_size, 100), device=device)
                generated_batch = g_net(noise_batch)
                predictions = d_net(generated_batch)
                target_gt = torch.ones((batch_size, 1), device=device)
                loss = loss_fn(predictions, target_gt)
                loss.backward()
                losses.append(loss.item())

                g_opt.step()

                g_opt.zero_grad()
                d_net.zero_grad()

            if cnt % 50 == 0:
                with torch.no_grad():
                    generated_batch = g_net(ref_noise_batch)
                    generated_batch = generated_batch.view(generated_batch.shape[0], 1, 28, 28)
                    new_real_batch = real_batch.view(real_batch.shape[0], 1, 28, 28)
                    composed = compose_imgs(generated_batch)
                    real_composed = compose_imgs(new_real_batch[:5])

                    if epoch % 1 == 0 and cnt == 0:
                        plt.imshow(np.vstack([np.repeat(real_composed, 3, axis=2), np.repeat(composed, 3, axis=2)]))
                        plt.show()

                        plt.plot(real_losses, 'r', label='d real loss')  # plotting t, a separately
                        plt.plot(fake_losses, 'b', label='d fake loss')  # plotting t, b separately
                        plt.plot(losses, 'g', label='g loss')  # plotting t, c separately
                        plt.legend()
                        plt.show()

                    cv.imwrite(os.path.join(debug_path, f'{epoch}_{cnt}.jpg'), cv.resize(composed, (0,0), fx=5, fy=5, interpolation=cv.INTER_NEAREST))
