import os


import git
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam


from .constants import LATENT_SPACE_DIM
from models.definitions.vanilla_gan_nets import DiscriminatorNet, GeneratorNet


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def get_mnist_dataset(dataset_path):
    # It's good to normalize the images to [-1, 1] range https://github.com/soumith/ganhacks
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])
    # This will download the MNIST the first time it is called
    return datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)


def get_mnist_data_loader(dataset_path, batch_size):
    mnist_dataset = get_mnist_dataset(dataset_path)
    mnist_data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return mnist_data_loader


def get_gaussian_latent_batch(batch_size, device):
    return torch.randn((batch_size, LATENT_SPACE_DIM), device=device)


def get_vanilla_nets(device):
    d_net = DiscriminatorNet().train().to(device)
    g_net = GeneratorNet().train().to(device)
    return d_net, g_net


# Tried SGD for the discriminator, had problems tweaking it - Adam simply works nicely but default lr 1e-3 won't work!
# I had to train discriminator more (4 to 1 schedule worked) to get it working with default lr, still got worse results.
# 0.0002 and 0.5, 0.999 are from the DCGAN paper it works here nicely!
def get_optimizers(d_net, g_net):
    d_opt = Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_opt = Adam(g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return d_opt, g_opt


def get_training_state(generator_net):
    training_state = {
        "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "state_dict": generator_net.state_dict()
    }
    return training_state


def print_training_info_to_console(training_config):
    print(f'Starting the GAN training.')
    print('*' * 80)
    print(f'Settings: num_epochs={training_config["num_epochs"]}, batch_size={training_config["batch_size"]}')
    print('*' * 80)

    if training_config["console_log_freq"]:
        print(f'Logging to console every {training_config["console_log_freq"]} batches.')
    else:
        print(f'Console logging disabled. Set console_log_freq if you want to use it.')

    print('')

    if training_config["debug_imagery_log_freq"]:
        print(f'Saving intermediate generator images to {training_config["debug_path"]} every {training_config["debug_imagery_log_freq"]} batches.')
    else:
        print(f'Generator intermediate image saving disabled. Set debug_imagery_log_freq you want to use it')

    print('')

    if training_config["checkpoint_freq"]:
        print(f'Saving checkpoint models to {training_config["checkpoints_path"]} every {training_config["checkpoint_freq"]} epochs.')
    else:
        print(f'Checkpoint models saving disabled. Set checkpoint_freq you want to use it')

    print('')

    if training_config['enable_tensorboard']:
        print('Tensorboard enabled. Logging generator and discriminator losses.')
        print('Run "tensorboard --logdir=runs" from your Anaconda (with conda env activated)')
        print('Open http://localhost:6006/ in your browser and you\'re ready to use tensorboard!')
    else:
        print('Tensorboard logging disabled.')
    print('*' * 80)

