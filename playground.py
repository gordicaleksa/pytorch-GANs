import os


import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


from models.definitions.vanilla_gan_nets import GeneratorNet
import utils.utils as utils


def understand_adversarial_loss():
    """Understand why we can use binary cross entropy as adversarial loss.

    It's currently setup so as to make discriminator's output close to 1 (we assume real images),
    but you can create fake_images_gt = torch.tensor(0.) and do a similar thing for fake images.

    How to use it:
        Read through the comments and analyze the console output.

    """
    adversarial_loss = nn.BCELoss()

    logits = [-10, -3, 0, 3, 10]  # Simulation of discriminator net's outputs before sigmoid activation

    # This setup the BCE loss into -log(x) (0. would set it as -log(1-x))
    real_images_gt = torch.tensor(1.)

    lr = 0.1  # learning rate

    for logit in logits:
        print('*' * 5)

        # Consider this as discriminator net's last layer's output
        # just before the sigmoid which converts it to probability.
        logit_tensor = torch.tensor(float(logit), requires_grad=True)
        print(f'logit value before optimization: {logit}')

        # Note: with_requires grad we force PyTorch to build the computational graph so that we can push the logit
        # towards values which will give us probability 1

        # Discriminator's output (probability that the image is real)
        prediction = nn.Sigmoid()(logit_tensor)
        print(f'discriminator net\'s output: {prediction}')

        # The closer the prediction is to 1 the lower the loss will be!
        # -log(prediction) <- for predictions close to 1 loss will be close to 0,
        # predictions close to 0 will cause the loss to go to "+ infinity".
        loss = adversarial_loss(prediction, real_images_gt)
        print(f'adversarial loss output: {loss}')

        loss.backward()  # calculate the gradient (sets the .grad field of the logit_tensor)
        # The closer the discriminator's prediction is to 1 the closer the loss will be to 0,
        # and the smaller this gradient will be, as there is no need to change logit,
        # because we accomplished what we wanted - to make prediction as close to 1 as possible.
        print(f'logit gradient {logit_tensor.grad.data}')

        # Effectively the biggest update will be made for logit -10.
        # Logit value -10 will cause the discriminator to output probability close to 0, which will give us huge loss
        # -log(0), which will cause big (negative) grad value which will then push the logit towards "+ infinity",
        # as that forces the discriminator to output the probability of 1. So -10 goes to ~ -9.9 in the first iteration.
        logit_tensor.data -= lr * logit_tensor.grad.data
        print(f'logit value after optimization {logit_tensor}')

        print('')


def generate_new_image():
    binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the model - load the weights and put the model into evaluation mode
    generator = GeneratorNet().to(device)
    training_state = torch.load(os.path.join(binaries_path, 'vanilla_generator_final.pth'))
    state_dict = training_state["state_dict"]
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()

    with torch.no_grad():
        latent_vector = utils.get_gaussian_latent_batch(1, device)
        generated_img = np.moveaxis(generator(latent_vector).to('cpu').numpy()[0], 0, 2)
        plt.imshow(np.repeat(generated_img, 3, axis=2)); plt.show()
        # utils.save_and_maybe_display_image(inference_config, stylized_img,
        #                                should_display=inference_config['should_not_display'])


if __name__ == "__main__":
    # understand_adversarial_loss()

    generate_new_image()


