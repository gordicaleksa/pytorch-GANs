import os


import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image


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


def postprocess_generated_img(generated_img_tensor):
    assert isinstance(generated_img_tensor, torch.Tensor), f'Expected PyTorch tensor but got {type(generated_img_tensor)}.'

    generated_img = np.repeat(np.moveaxis(generated_img_tensor.to('cpu').numpy()[0], 0, 2), 3, axis=2)
    generated_img -= np.min(generated_img)
    generated_img /= np.max(generated_img)
    return generated_img


def generate_random(generator, device):
    with torch.no_grad():
        latent_vector = utils.get_gaussian_latent_batch(1, device)
        generated_img = postprocess_generated_img(generator(latent_vector))
    return generated_img, latent_vector.to('cpu').numpy()[0]


def generate_for_latent_vector(generator, device, latent_vector):
    assert isinstance(latent_vector, np.ndarray), f'Expected latent vector to be numpy array but got {type(latent_vector)}.'

    latent_vector_tensor = torch.unsqueeze(torch.tensor(latent_vector, device=device), dim=0)
    with torch.no_grad():
        return postprocess_generated_img(generator(latent_vector_tensor))


def generate_new_image(interpolation_mode=True):
    binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    model_path = os.path.join(binaries_path, 'vanilla_generator_final.pth')
    assert os.path.exists(model_path), f'Could not find the model {model_path}. You first need to train your generator.'

    generated_imgs_path = os.path.join(os.path.dirname(__file__), 'data', 'generated')
    interpolated_imgs_path = os.path.join(os.path.dirname(__file__), 'data', 'interpolated')
    os.makedirs(generated_imgs_path, exist_ok=True)
    os.makedirs(interpolated_imgs_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the model - load the weights and put the model into evaluation mode
    generator = GeneratorNet().to(device)
    generator.load_state_dict(torch.load(model_path)["state_dict"], strict=True)
    generator.eval()

    if interpolation_mode:
        latent_vector_a, latent_vector_b = [None, None]
        found_good_vectors_flag = False
        while not found_good_vectors_flag:
            generated_img, latent_vector = generate_random(generator, device)
            plt.imshow(generated_img); plt.show()
            user_input = input("Do you like this generated image? [y for yes]:")
            if user_input == 'y':
                if latent_vector_a is None:
                    latent_vector_a = latent_vector
                    print('Saved the first latent vector.')
                elif latent_vector_b is None:
                    latent_vector_b = latent_vector
                    print('Saved the second latent vector.')
                    found_good_vectors_flag = True
            else:
                print('Well lets generate a new one!')
                continue

        print('Lets do some interpolation!')
        interpolation_resolution = 47  # number of images between the vectors a and b
        num_imgs = interpolation_resolution + 2  # + 2 so that we include a and b
        diff_vector = latent_vector_b - latent_vector_a
        generated_imgs = []
        for i in range(num_imgs):
            scale_coeff = i / (num_imgs - 1)  # goes from 0. to 1.
            current_latent_vector = latent_vector_a + scale_coeff * diff_vector
            generated_img = generate_for_latent_vector(generator, device, current_latent_vector)
            print(f'Generated image [{i+1}/{num_imgs}].')
            # plt.imshow(generated_img); plt.show()
            generated_imgs.append(np.moveaxis(generated_img, 2, 0))
        interpolated_block_img = torch.from_numpy(np.stack(generated_imgs))
        interpolated_block_img = nn.Upsample(scale_factor=2.5, mode='nearest')(interpolated_block_img)
        save_image(interpolated_block_img, os.path.join(interpolated_imgs_path, f'interpolated_block.jpg'), nrow=int(np.sqrt(num_imgs)))
    else:
        generated_img, _ = generate_random(generator, device)
        utils.save_and_maybe_display_image(generated_imgs_path, generated_img, should_display=False)


if __name__ == "__main__":
    # understand_adversarial_loss()

    generate_new_image(interpolation_mode=True)


