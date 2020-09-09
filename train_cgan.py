import os
import argparse
import time
import random


import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter


import utils.utils as utils
from utils.constants import *


def train_cgan(training_config):
    writer = SummaryWriter()  # (tensorboard) writer will output to ./runs/ directory by default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    # Prepare MNIST data loader (it will download MNIST the first time you run it)
    mnist_data_loader = utils.get_mnist_data_loader(DATA_DIR_PATH, training_config['batch_size'])

    # Fetch feed-forward nets (place them on GPU if present) and optimizers which will tweak their weights
    discriminator_net, generator_net = utils.get_gan(device, GANType.CGAN.name)
    discriminator_opt, generator_opt = utils.get_optimizers(discriminator_net, generator_net)

    # 1s will configure BCELoss into -log(x) whereas 0s will configure it to -log(1-x)
    # So that means we can effectively use binary cross-entropy loss to achieve adversarial loss!
    adversarial_loss = nn.BCELoss()
    real_images_gt = torch.ones((training_config['batch_size'], 1), device=device)
    fake_images_gt = torch.zeros((training_config['batch_size'], 1), device=device)

    # For logging purposes
    ref_batch_size = MNIST_NUM_CLASSES**2  # We'll create a grid 10x10 where each column is a single digit
    ref_noise_batch = utils.get_gaussian_latent_batch(ref_batch_size, device)  # Track G's quality during training

    # We'll generate exactly this grid of 10x10 (each digit in a separate column and 10 instances) for easier debugging
    ref_labels = torch.tensor(np.array([digit for _ in range(MNIST_NUM_CLASSES) for digit in range(MNIST_NUM_CLASSES)]), dtype=torch.int64)
    ref_labels_one_hot = torch.nn.functional.one_hot(ref_labels, MNIST_NUM_CLASSES).type(torch.FloatTensor).to(device)

    discriminator_loss_values = []
    generator_loss_values = []
    img_cnt = 0

    ts = time.time()  # start measuring time

    # GAN training loop
    utils.print_training_info_to_console(training_config)
    for epoch in range(training_config['num_epochs']):
        for batch_idx, (real_images, labels) in enumerate(mnist_data_loader):

            labels_one_hot = torch.nn.functional.one_hot(labels, MNIST_NUM_CLASSES).type(torch.FloatTensor).to(device)
            real_images = real_images.to(device)  # Place imagery on GPU (if present)

            #
            # Train discriminator: maximize V = log(D(x)) + log(1-D(G(z))) or equivalently minimize -V
            # Note: D = discriminator, x = real images, G = generator, z = latent Gaussian vectors, G(z) = fake images
            #

            # Zero out .grad variables in discriminator network (otherwise we would have corrupt results)
            discriminator_opt.zero_grad()

            # -log(D(x)) <- we minimize this by making D(x)/discriminator_net(real_images) as close to 1 as possible
            real_discriminator_loss = adversarial_loss(discriminator_net(real_images, labels_one_hot), real_images_gt)

            # G(z) | G == generator_net and z == utils.get_gaussian_latent_batch(batch_size, device)
            fake_images = generator_net(utils.get_gaussian_latent_batch(training_config['batch_size'], device), labels_one_hot)
            # D(G(z)), we call detach() so that we don't calculate gradients for the generator during backward()
            fake_images_predictions = discriminator_net(fake_images.detach(), labels_one_hot)
            # -log(1 - D(G(z))) <- we minimize this by making D(G(z)) as close to 0 as possible
            fake_discriminator_loss = adversarial_loss(fake_images_predictions, fake_images_gt)

            discriminator_loss = real_discriminator_loss + fake_discriminator_loss
            discriminator_loss.backward()  # this will populate .grad vars in the discriminator net
            discriminator_opt.step()  # perform D weights update according to optimizer's strategy

            #
            # Train generator: minimize V1 = log(1-D(G(z))) or equivalently maximize V2 = log(D(G(z))) (or min of -V2)
            # The original expression (V1) had problems with diminishing gradients for G when D is too good.
            #

            # if you want to cause mode collapse probably the easiest way to do that would be to add "for i in range(n)"
            # here (simply train G more frequent than D), n = 10 worked for me other values will also work - experiment.

            # Zero out .grad variables in discriminator network (otherwise we would have corrupt results)
            generator_opt.zero_grad()

            # D(G(z)) (see above for explanations)
            generated_images_predictions = discriminator_net(generator_net(utils.get_gaussian_latent_batch(training_config['batch_size'], device), labels_one_hot), labels_one_hot)
            # By placing real_images_gt here we minimize -log(D(G(z))) which happens when D approaches 1
            # i.e. we're tricking D into thinking that these generated images are real!
            generator_loss = adversarial_loss(generated_images_predictions, real_images_gt)

            generator_loss.backward()  # this will populate .grad vars in the G net (also in D but we won't use those)
            generator_opt.step()  # perform G weights update according to optimizer's strategy

            #
            # Logging and checkpoint creation
            #

            generator_loss_values.append(generator_loss.item())
            discriminator_loss_values.append(discriminator_loss.item())

            if training_config['enable_tensorboard']:
                writer.add_scalars('losses/g-and-d', {'g': generator_loss.item(), 'd': discriminator_loss.item()}, len(mnist_data_loader) * epoch + batch_idx + 1)
                # Save debug imagery to tensorboard also (some redundancy but it may be more beginner-friendly)
                if training_config['debug_imagery_log_freq'] is not None and batch_idx % training_config['debug_imagery_log_freq'] == 0:
                    with torch.no_grad():
                        log_generated_images = generator_net(ref_noise_batch, ref_labels_one_hot)
                        log_generated_images_resized = nn.Upsample(scale_factor=1.5, mode='nearest')(log_generated_images)
                        intermediate_imagery_grid = make_grid(log_generated_images_resized, nrow=int(np.sqrt(ref_batch_size)), normalize=True)
                        writer.add_image('intermediate generated imagery', intermediate_imagery_grid, len(mnist_data_loader) * epoch + batch_idx + 1)

            if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                print(f'GAN training: time elapsed= {(time.time() - ts):.2f} [s] | epoch={epoch + 1} | batch= [{batch_idx + 1}/{len(mnist_data_loader)}]')

            # Save intermediate generator images (more convenient like this than through tensorboard)
            if training_config['debug_imagery_log_freq'] is not None and batch_idx % training_config['debug_imagery_log_freq'] == 0:
                with torch.no_grad():
                    log_generated_images = generator_net(ref_noise_batch, ref_labels_one_hot)
                    log_generated_images_resized = nn.Upsample(scale_factor=1.5, mode='nearest')(log_generated_images)
                    save_image(log_generated_images_resized, os.path.join(training_config['debug_path'], f'{str(img_cnt).zfill(6)}.jpg'), nrow=int(np.sqrt(ref_batch_size)), normalize=True)
                    img_cnt += 1

            # Save generator checkpoint
            if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0 and batch_idx == 0:
                ckpt_model_name = f"cgan_ckpt_epoch_{epoch + 1}_batch_{batch_idx + 1}.pth"
                torch.save(utils.get_training_state(generator_net, GANType.CGAN.name), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

    # Save the latest generator in the binaries directory
    torch.save(utils.get_training_state(generator_net, GANType.CGAN.name), os.path.join(BINARIES_PATH, utils.get_available_binary_name(GANType.CGAN)))


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    debug_path = os.path.join(DATA_DIR_PATH, 'debug_imagery')
    os.makedirs(debug_path, exist_ok=True)

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, help="height of content and style images", default=50)
    parser.add_argument("--batch_size", type=int, help="height of content and style images", default=128)

    # logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging (D and G loss)", default=True)
    parser.add_argument("--debug_imagery_log_freq", type=int, help="log generator images during training (batch) freq", default=100)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=100)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=5)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['debug_path'] = debug_path

    # train GAN model
    train_cgan(training_config)

