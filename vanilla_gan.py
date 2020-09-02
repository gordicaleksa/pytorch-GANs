import os
import copy
import argparse


import cv2 as cv
import torch
from torch import nn

import utils.utils as utils
from models.definitions.vanilla_gan_nets import get_vanilla_nets

# todo: create a video of the debug imagery
# todo: create fine step interpolation imagery and make a video out of those
# todo: try out save_image from torchvision.utils
# todo: force mode collapse and add settings and results to readme

# todo: modify archs and see how it behaves
# todo: Try 1D normalization in generator and discriminator (like in DCGAN)
# todo: use ReLU in generator instead of leaky, compare leaky vs non-leaky ReLU
# todo: do view inside the archs themself

# todo: try changing Adams beta1 to 0.5 (like in DCGAN)
# todo: try out SGD for discriminator net


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    mnist_path = data_dir
    debug_path = os.path.join(os.path.dirname(__file__), 'data', 'debug_dir_dstep1_betasbrt')
    os.makedirs(debug_path, exist_ok=True)

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="height of content and style images", default=128)
    parser.add_argument("--num_epochs", type=int, help="height of content and style images", default=50)
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    # Prepare MNIST data loader (it will download MNIST the first time you run it)
    mnist_data_loader = utils.get_mnist_data_loader(mnist_path, batch_size)

    d_net, g_net = get_vanilla_nets(device)
    d_opt, g_opt = utils.prepare_optimizers(d_net, g_net)

    adversarial_loss = nn.BCELoss()

    ref_noise_batch = torch.randn((5, 100), device=device)

    d_losses = []
    g_losses = []

    real_images_gt = torch.ones((batch_size, 1), device=device)
    fake_images_gt = torch.zeros((batch_size, 1), device=device)

    for epoch in range(num_epochs):
        for batch_idx, (real_batch, _) in enumerate(mnist_data_loader):

            if batch_idx % 100 == 0:
                print(f'Training. Epoch = {epoch} batch = {batch_idx}.')

            real_batch = real_batch.to(device)

            #
            # Train discriminator
            #

            d_net.zero_grad()

            # Train discriminator net
            real_predictions = d_net(real_batch)
            real_loss = adversarial_loss(real_predictions, real_images_gt)

            noise_batch = utils.get_latent_batch(batch_size, device)
            fake_batch = g_net(noise_batch)
            fake_predictions = d_net(fake_batch.detach())
            fake_loss = adversarial_loss(fake_predictions, fake_images_gt)
            d_loss = real_loss + fake_loss

            d_loss.backward()
            d_losses.append(d_loss.item())

            d_opt.step()

            #
            # Train generator net
            #

            g_opt.zero_grad()

            noise_batch = utils.get_latent_batch(batch_size, device)
            generated_batch = g_net(noise_batch)
            predictions = d_net(generated_batch)
            g_loss = adversarial_loss(predictions, real_images_gt)

            g_loss.backward()
            g_losses.append(g_loss.item())

            g_opt.step()

            if batch_idx % 50 == 0:
                with torch.no_grad():
                    generated_batch = g_net(ref_noise_batch)
                    generated_batch = generated_batch.view(generated_batch.shape[0], 1, 28, 28)
                    new_real_batch = real_batch.view(real_batch.shape[0], 1, 28, 28)
                    composed = utils.compose_imgs(generated_batch)
                    real_composed = utils.compose_imgs(new_real_batch[:5])

                    # if epoch % 1 == 0 and cnt == 0:
                    #     plt.imshow(np.vstack([np.repeat(real_composed, 3, axis=2), np.repeat(composed, 3, axis=2)]))
                    #     plt.show()
                    #
                    #     plt.plot(real_losses, 'r', label='d real loss')  # plotting t, a separately
                    #     plt.plot(fake_losses, 'b', label='d fake loss')  # plotting t, b separately
                    #     plt.plot(losses, 'g', label='g loss')  # plotting t, c separately
                    #     plt.legend()
                    #     plt.show()

                    cv.imwrite(os.path.join(debug_path, f'{epoch}_{batch_idx}.jpg'), cv.resize(composed, (0, 0), fx=5, fy=5, interpolation=cv.INTER_NEAREST))
