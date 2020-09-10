import os
import re
import zipfile


import git
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.hub import download_url_to_file


from .constants import *
from models.definitions.vanilla_gan import DiscriminatorNet, GeneratorNet
from models.definitions.conditional_gan import ConditionalDiscriminatorNet, ConditionalGeneratorNet
from models.definitions.dcgan import ConvolutionalDiscriminativeNet, ConvolutionalGenerativeNet


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


def save_and_maybe_display_image(dump_dir, dump_img, out_res=(256, 256), should_display=False):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'

    # step1: get next valid image name
    dump_img_name = get_available_file_name(dump_dir)

    # step2: convert to uint8 format
    if dump_img.dtype != np.uint8:
        dump_img = (dump_img*255).astype(np.uint8)

    # step3: write image to the file system
    cv.imwrite(os.path.join(dump_dir, dump_img_name), cv.resize(dump_img[:, :, ::-1], out_res, interpolation=cv.INTER_NEAREST))  # ::-1 because opencv expects BGR (and not RGB) format...

    # step4: maybe display part of the function
    if should_display:
        plt.imshow(dump_img)
        plt.show()


def get_available_file_name(input_dir):
    def valid_frame_name(str):
        pattern = re.compile(r'[0-9]{6}\.jpg')  # regex, examples it covers: 000000.jpg or 923492.jpg, etc.
        return re.fullmatch(pattern, str) is not None

    valid_frames = list(filter(valid_frame_name, os.listdir(input_dir)))
    if len(valid_frames) > 0:
        # Images are saved in the <xxxxxx>.jpg format we find the biggest such <xxxxxx> number and increment by 1
        last_img_name = sorted(valid_frames)[-1]
        new_prefix = int(last_img_name.split('.')[0]) + 1  # increment by 1
        return f'{str(new_prefix).zfill(6)}.jpg'
    else:
        return '000000.jpg'


def get_available_binary_name(gan_type_enum=GANType.VANILLA):
    def valid_binary_name(binary_name):
        # First time you see raw f-string? Don't worry the only trick is to double the brackets.
        pattern = re.compile(rf'{gan_type_enum.name}_[0-9]{{6}}\.pth')
        return re.fullmatch(pattern, binary_name) is not None

    prefix = gan_type_enum.name
    # Just list the existing binaries so that we don't overwrite them but write to a new one
    valid_binary_names = list(filter(valid_binary_name, os.listdir(BINARIES_PATH)))
    if len(valid_binary_names) > 0:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split('.')[0][-6:]) + 1  # increment by 1
        return f'{prefix}_{str(new_suffix).zfill(6)}.pth'
    else:
        return f'{prefix}_000000.pth'


def get_gan_data_transform():
    # It's good to normalize the images to [-1, 1] range https://github.com/soumith/ganhacks
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])
    return transform


def get_mnist_dataset():
    # This will download the MNIST the first time it is called
    return datasets.MNIST(root=DATA_DIR_PATH, train=True, download=True, transform=get_gan_data_transform())


def get_mnist_data_loader(batch_size):
    mnist_dataset = get_mnist_dataset()
    mnist_data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return mnist_data_loader


def download_and_prepare_celeba(celeba_path):
    celeba_url = r'https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip'

    # Step1: Download the resource to local filesystem
    print('*' * 50)
    print(f'Downloading {celeba_url}.')
    print('This may take a while the first time, the zip file has 240 MBs.')
    print('*' * 50)

    resource_tmp_path = celeba_path + '.zip'
    download_url_to_file(celeba_url, resource_tmp_path)

    # Step2: Unzip the resource
    print(f'Started unzipping. Go and take a cup of coffe.')
    with zipfile.ZipFile(resource_tmp_path) as zf:
        os.makedirs(celeba_path, exist_ok=True)
        zf.extractall(path=celeba_path)
    print(f'Unzipping to: {celeba_path} finished.')

    # Step3: Remove the temporary resource file
    os.remove(resource_tmp_path)
    print(f'Removing tmp file {resource_tmp_path}.')

    # Step4: Prepare the dataset into a suitable format for PyTorch's ImageFolder
    # todo: prepare for ImageFolder


def get_celeba_data_loader(batch_size):
    celeba_path = os.path.join(DATA_DIR_PATH, 'CelebA')
    if not os.path.exists(celeba_path):  # We'll have to do this only 1 time, I promise.
        download_and_prepare_celeba(celeba_path)

    celeba_dataset = ImageFolder(celeba_path, transform=get_gan_data_transform())
    celeba_data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return celeba_data_loader


def get_gaussian_latent_batch(batch_size, device):
    return torch.randn((batch_size, LATENT_SPACE_DIM), device=device)


def get_gan(device, gan_type_name):
    assert gan_type_name in [gan_type.name for gan_type in GANType], f'Unknown GAN type = {gan_type_name}.'

    if gan_type_name == GANType.VANILLA.name:
        d_net = DiscriminatorNet().train().to(device)
        g_net = GeneratorNet().train().to(device)
    elif gan_type_name == GANType.CGAN.name:
        d_net = ConditionalDiscriminatorNet().train().to(device)
        g_net = ConditionalGeneratorNet().train().to(device)
    elif gan_type_name == GANType.DCGAN.name:
        d_net = ConvolutionalDiscriminativeNet().train().to(device)
        g_net = ConvolutionalGenerativeNet().train().to(device)
    else:
        raise Exception(f'GAN type {gan_type_name} not yet supported.')

    return d_net, g_net


# Tried SGD for the discriminator, had problems tweaking it - Adam simply works nicely but default lr 1e-3 won't work!
# I had to train discriminator more (4 to 1 schedule worked) to get it working with default lr, still got worse results.
# 0.0002 and 0.5, 0.999 are from the DCGAN paper it works here nicely!
def get_optimizers(d_net, g_net):
    d_opt = Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_opt = Adam(g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    return d_opt, g_opt


def get_training_state(generator_net, gan_type_name):
    training_state = {
        "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "state_dict": generator_net.state_dict(),
        "gan_type": gan_type_name
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
        print(f'Saving checkpoint models to {CHECKPOINTS_PATH} every {training_config["checkpoint_freq"]} epochs.')
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

