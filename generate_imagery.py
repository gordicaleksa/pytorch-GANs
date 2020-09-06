import os


import torch
from torch import nn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np


from models.definitions.vanilla_gan_nets import GeneratorNet
import utils.utils as utils
from utils.constants import *


def postprocess_generated_img(generated_img_tensor):
    assert isinstance(generated_img_tensor, torch.Tensor), f'Expected PyTorch tensor but got {type(generated_img_tensor)}.'

    # Move the tensor from GPU to CPU, convert to numpy array, extract 0th batch, move the image channel
    # from 0th to 2nd position (CHW -> HWC), repeat grayscale image 3 times to get RGB image (as we train on MNIST)
    generated_img = np.repeat(np.moveaxis(generated_img_tensor.to('cpu').numpy()[0], 0, 2), 3, axis=2)

    # Imagery is in the range [-1, 1] (generator has tanh as the output activation) move it into [0, 1] range
    generated_img -= np.min(generated_img)
    generated_img /= np.max(generated_img)

    return generated_img


def generate_from_random_latent_vector(generator):
    with torch.no_grad():
        latent_vector = utils.get_gaussian_latent_batch(1, next(generator.parameters()).device)
        generated_img = postprocess_generated_img(generator(latent_vector))
    return generated_img, latent_vector.to('cpu').numpy()[0]


def generate_from_specified_numpy_latent_vector(generator, latent_vector):
    assert isinstance(latent_vector, np.ndarray), f'Expected latent vector to be numpy array but got {type(latent_vector)}.'

    with torch.no_grad():
        latent_vector_tensor = torch.unsqueeze(torch.tensor(latent_vector, device=next(generator.parameters()).device), dim=0)
        return postprocess_generated_img(generator(latent_vector_tensor))


def generate_new_images(interpolation_mode=True, should_display=True):
    """ Generate imagery using pre-trained generator (using vanilla_generator_final.pth by default)

    Args:
        interpolation_mode (bool):
            if True linearly interpolate between the 2 chosen latent vectors,
            and generate a spectrum of images between those 2, if False generate a single image from a random vector

        should_display (bool): Display the generated images before saving them

    """

    model_path = os.path.join(BINARIES_PATH, 'vanilla_generator_final.pth')
    assert os.path.exists(model_path), f'Could not find the model {model_path}. You first need to train your generator.'

    generated_imgs_path = os.path.join(DATA_DIR_PATH, 'generated')
    interpolated_imgs_path = os.path.join(DATA_DIR_PATH, 'interpolated')
    os.makedirs(generated_imgs_path, exist_ok=True)
    os.makedirs(interpolated_imgs_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the model - load the weights and put the model into evaluation mode
    generator = GeneratorNet().to(device)
    generator.load_state_dict(torch.load(model_path)["state_dict"], strict=True)
    generator.eval()

    # Pick 2 images you like between which you'd like to linearly interpolate (by typing 'y' into console)
    if interpolation_mode:
        latent_vector_a, latent_vector_b = [None, None]
        found_good_vectors_flag = False

        # Loop until the user picked the 2 images he/she likes.
        while not found_good_vectors_flag:
            generated_img, latent_vector = generate_from_random_latent_vector(generator)
            plt.imshow(generated_img); plt.title('Do you like this image?'); plt.show()
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

        # Spherical would probably do better, check out: https://github.com/soumith/ganhacks
        print('Lets do some linear interpolation!')
        interpolation_resolution = 47  # number of images between the vectors a and b
        num_interpolated_imgs = interpolation_resolution + 2  # + 2 so that we include a and b

        diff_vector = latent_vector_b - latent_vector_a
        generated_imgs = []
        for i in range(num_interpolated_imgs):
            scale_coeff = i / (num_interpolated_imgs - 1)  # goes from 0. to 1.
            current_latent_vector = latent_vector_a + scale_coeff * diff_vector  # linear interpolation step
            generated_img = generate_from_specified_numpy_latent_vector(generator, current_latent_vector)

            print(f'Generated image [{i+1}/{num_interpolated_imgs}].')
            if should_display:
                plt.imshow(generated_img); plt.show()

            # Move from channel last to channel first (CHW->HWC), PyTorch's save_image function expects BCHW format
            generated_imgs.append(np.moveaxis(generated_img, 2, 0))

        interpolated_block_img = torch.from_numpy(np.stack(generated_imgs))
        interpolated_block_img = nn.Upsample(scale_factor=2.5, mode='nearest')(interpolated_block_img)
        save_image(interpolated_block_img, os.path.join(interpolated_imgs_path, f'interpolated_block.jpg'), nrow=int(np.sqrt(num_interpolated_imgs)))
    else:
        generated_img, _ = generate_from_random_latent_vector(generator)
        utils.save_and_maybe_display_image(generated_imgs_path, generated_img, should_display=should_display)


if __name__ == "__main__":
    generate_new_images(interpolation_mode=True, should_display=True)

