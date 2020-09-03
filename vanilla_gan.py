import os
import argparse


import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image


import utils.utils as utils

# todo: create a video of the debug imagery
# todo: create fine step interpolation imagery and make a video out of those
# todo: force mode collapse and add settings and results to readme

# todo: modify archs and see how it behaves
# todo: Try 1D normalization in generator and discriminator (like in DCGAN)
# todo: use ReLU in generator instead of leaky, compare leaky vs non-leaky ReLU

if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    mnist_dataset_path = data_dir
    debug_path = os.path.join(data_dir, 'intermediate_imagery')
    os.makedirs(model_binaries_path, exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)

    console_log_freq = 100
    ref_generated_images_log_freq = 100
    model_checkpoint_log_freq_per_epoch = 5
    ref_batch_size = 16

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="height of content and style images", default=128)
    parser.add_argument("--num_epochs", type=int, help="height of content and style images", default=50)
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    # Prepare MNIST data loader (it will download MNIST the first time you run it)
    mnist_data_loader = utils.get_mnist_data_loader(mnist_dataset_path, batch_size)

    # Fetch feed-forward nets (placed on GPU if present) and optimizers which control their weights
    discriminator_net, generator_net = utils.get_vanilla_nets(device)
    discriminator_opt, generator_opt = utils.prepare_optimizers(discriminator_net, generator_net)

    # 1s will configure BCELoss into -log(x) whereas 0s will configure it to -log(1-x)
    # So that means we can effectively use binary cross-entropy loss to achieve adversarial loss!
    adversarial_loss = nn.BCELoss()
    real_images_gt = torch.ones((batch_size, 1), device=device)
    fake_images_gt = torch.zeros((batch_size, 1), device=device)

    # Logging purposes
    ref_noise_batch = utils.get_gaussian_latent_batch(ref_batch_size, device)  # Track G's quality during training
    discriminator_loss_values = []
    generator_loss_values = []

    # GAN training loop
    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(mnist_data_loader):

            real_images = real_images.to(device)  # Place imagery on GPU (if present)

            #
            # Train discriminator: maximize V = log(D(x)) + log(1-D(G(z))) or equivalently minimize -V
            # Note: D = discriminator, x = real images, G = generator, z = latent Gaussian vectors, G(z) = fake images
            #

            # Zero out .grad variables in discriminator network (otherwise we would have corrupt results)
            discriminator_opt.zero_grad()

            # -log(D(x)) <- we minimize this by making D(x)/discriminator_net(real_images) as close to 1 as possible
            real_discriminator_loss = adversarial_loss(discriminator_net(real_images), real_images_gt)

            # G(z) | G == generator_net and z == utils.get_gaussian_latent_batch(batch_size, device)
            fake_images = generator_net(utils.get_gaussian_latent_batch(batch_size, device))
            # D(G(z)), we call detach() so that we don't calculate gradients for the generator during backward()
            fake_images_predictions = discriminator_net(fake_images.detach())
            # -log(1 - D(G(z))) <- we minimize this by making D(G(z)) as close to 0 as possible
            fake_discriminator_loss = adversarial_loss(fake_images_predictions, fake_images_gt)

            discriminator_loss = real_discriminator_loss + fake_discriminator_loss
            discriminator_loss.backward()  # this will populate .grad vars in the discriminator net
            discriminator_opt.step()  # perform D weights update according to optimizer's strategy

            #
            # Train generator: minimize V1 = log(1-D(G(z))) or equivalently maximize V2 = log(D(G(z))) (or min of -V2)
            # The original expression (V1) had problems with diminishing gradients for G when D is too good.
            #

            # Zero out .grad variables in discriminator network (otherwise we would have corrupt results)
            generator_opt.zero_grad()

            # D(G(z)) (see above for explanations)
            generated_images_predictions = discriminator_net(generator_net(utils.get_gaussian_latent_batch(batch_size, device)))
            # By placing real_images_gt here we minimize -log(D(G(z))) which happens when D approaches 1
            # i.e. we're tricking D into thinking that these generated images are real!
            generator_loss = adversarial_loss(generated_images_predictions, real_images_gt)

            generator_loss.backward()  # this will populate .grad vars in the G net (also in D but we won't use those)
            generator_opt.step()  # perform G weights update according to optimizer's strategy

            #
            # Logging and checkpoint creation
            #
            # todo: add checkpoint model saving
            # todo: add tensorboard loss logging
            generator_loss_values.append(generator_loss.item())
            discriminator_loss_values.append(discriminator_loss.item())

            if batch_idx % console_log_freq == 0:
                print(f'Training GANs, epoch = {epoch} | batch = {batch_idx}.')

            #     plt.plot(real_losses, 'r', label='d real loss')  # plotting t, a separately
            #     plt.plot(fake_losses, 'b', label='d fake loss')  # plotting t, b separately
            #     plt.plot(losses, 'g', label='g loss')  # plotting t, c separately
            #     plt.legend()
            #     plt.show()

            if batch_idx % ref_generated_images_log_freq == 0:
                with torch.no_grad():
                    log_generated_images = generator_net(ref_noise_batch)
                    log_generated_images_resized = nn.Upsample(scale_factor=2.5, mode='nearest')(log_generated_images)
                    log_grid = save_image(log_generated_images_resized, os.path.join(debug_path, f'{epoch}_{batch_idx}.jpg'), nrow=int(np.sqrt(ref_batch_size)), normalize=True)
