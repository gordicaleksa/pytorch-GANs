import os
import shutil


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


def linear_interpolation(t, p0, p1):
    return p0 + t * (p1 - p0)


def spherical_interpolation(t, p0, p1):
    """ Spherical interpolation (slerp) formula: https://en.wikipedia.org/wiki/Slerp

    Found inspiration here: https://github.com/soumith/ganhacks
    but I didn't get any improvement using it compared to linear interpolation.

    Args:
        t (float): has [0, 1] range
        p0 (numpy array): First n-dimensional vector
        p1 (numpy array): Second n-dimensional vector

    Result:
        Returns spherically interpolated vector.

    """
    if t <= 0:
        return p0
    elif t >= 1:
        return p1
    elif np.allclose(p0, p1):
        return p0

    # Convert p0 and p1 to unit vectors and find the angle between them (omega)
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    sin_omega = np.sin(omega)  # syntactic sugar
    return np.sin((1.0 - t) * omega) / sin_omega * p0 + np.sin(t * omega) / sin_omega * p1


def generate_new_images(model_name, interpolation_mode=True, slerp=True, a=None, b=None, should_display=True):
    """ Generate imagery using pre-trained generator (using vanilla_generator_000000.pth by default)

    Args:
        model_name (str): model name you want to use (default lookup location is BINARIES_PATH).
        a, b (numpy arrays): latent vectors, if set to None you'll be prompted to choose images you like,
         and use corresponding latent vectors instead.
        interpolation_mode (bool): if True interpolate between the 2 chosen latent vectors,
        and generate a spectrum of images between those 2, if False generate a single image from a random vector.
        slerp (bool): if True use spherical interpolation otherwise use linear interpolation.
        should_display (bool): Display the generated images before saving them.

    """

    model_path = os.path.join(BINARIES_PATH, model_name)
    assert os.path.exists(model_path), f'Could not find the model {model_path}. You first need to train your generator.'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the model - load the weights and put the model into evaluation mode
    generator = GeneratorNet().to(device)
    generator.load_state_dict(torch.load(model_path)["state_dict"], strict=True)
    generator.eval()

    # Pick 2 images you like between which you'd like to interpolate (by typing 'y' into console)
    if interpolation_mode:
        interpolation_name = "spherical" if slerp else "linear"
        interpolation_fn = spherical_interpolation if slerp else linear_interpolation

        grid_interpolated_imgs_path = os.path.join(DATA_DIR_PATH, 'interpolated_imagery')  # combined results dir
        decomposed_interpolated_imgs_path = os.path.join(grid_interpolated_imgs_path, f'tmp_{interpolation_name}_dump')  # dump separate results
        if os.path.exists(decomposed_interpolated_imgs_path):
            shutil.rmtree(decomposed_interpolated_imgs_path)
        os.makedirs(grid_interpolated_imgs_path, exist_ok=True)
        os.makedirs(decomposed_interpolated_imgs_path, exist_ok=True)

        latent_vector_a, latent_vector_b = [None, None]

        # If a and b were not specified loop until the user picked the 2 images he/she likes.
        found_good_vectors_flag = False
        if a is None or b is None:
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
        else:
            print('Skip latent vectors selection section and use cached ones.')
            latent_vector_a, latent_vector_b = [a, b]

        # Cache latent vectors
        if a is None or b is None:
            np.save(os.path.join(grid_interpolated_imgs_path, 'a.npy'), latent_vector_a)
            np.save(os.path.join(grid_interpolated_imgs_path, 'b.npy'), latent_vector_b)

        print(f'Lets do some {interpolation_name} interpolation!')
        interpolation_resolution = 47  # number of images between the vectors a and b
        num_interpolated_imgs = interpolation_resolution + 2  # + 2 so that we include a and b

        generated_imgs = []
        for i in range(num_interpolated_imgs):
            t = i / (num_interpolated_imgs - 1)  # goes from 0. to 1.
            current_latent_vector = interpolation_fn(t, latent_vector_a, latent_vector_b)
            generated_img = generate_from_specified_numpy_latent_vector(generator, current_latent_vector)

            print(f'Generated image [{i+1}/{num_interpolated_imgs}].')
            utils.save_and_maybe_display_image(decomposed_interpolated_imgs_path, generated_img, should_display=should_display)

            # Move from channel last to channel first (CHW->HWC), PyTorch's save_image function expects BCHW format
            generated_imgs.append(np.moveaxis(generated_img, 2, 0))

        interpolated_block_img = torch.from_numpy(np.stack(generated_imgs))
        interpolated_block_img = nn.Upsample(scale_factor=2.5, mode='nearest')(interpolated_block_img)
        save_image(interpolated_block_img, os.path.join(grid_interpolated_imgs_path, utils.get_available_file_name(grid_interpolated_imgs_path)), nrow=int(np.sqrt(num_interpolated_imgs)))
    else:
        generated_imgs_path = os.path.join(DATA_DIR_PATH, 'generated_imagery')
        os.makedirs(generated_imgs_path, exist_ok=True)

        generated_img, _ = generate_from_random_latent_vector(generator)
        utils.save_and_maybe_display_image(generated_imgs_path, generated_img, should_display=should_display)


if __name__ == "__main__":
    # The first time you start generation in the interpolation mode it will cache a and b
    a_path = os.path.join(DATA_DIR_PATH, 'interpolated_imagery', 'a.npy')
    b_path = os.path.join(DATA_DIR_PATH, 'interpolated_imagery', 'b.npy')
    a = np.load(a_path) if os.path.exists(a_path) else None
    b = np.load(b_path) if os.path.exists(b_path) else None

    model_name = r'vanilla_generator_000000.pth'

    generate_new_images(model_name, interpolation_mode=True, slerp=True, a=a, b=b, should_display=False)

