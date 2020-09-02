import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam


from .constants import LATENT_SPACE_DIM


def get_mnist_dataset(dataset_path):
    # It's good to normalize the images to [-1, 1] range https://github.com/soumith/ganhacks
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])
    return datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)


def get_mnist_data_loader(dataset_path, batch_size):
    mnist_dataset = get_mnist_dataset(dataset_path)
    mnist_data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return mnist_data_loader


def get_latent_batch(batch_size, device):
    return torch.randn((batch_size, LATENT_SPACE_DIM), device=device)


def plot_single_img_from_tensor_batch(batch):
    img = batch[0].numpy()
    img = np.repeat(np.moveaxis(img, 0, 2), 3, axis=2)
    print(img.shape, np.min(img), np.max(img))
    plt.imshow(img)
    plt.show()


def prepare_optimizers(d_net, g_net):
    d_opt = Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_opt = Adam(g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
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