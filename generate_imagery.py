import os
import shutil
import argparse


import torch
from torch import nn
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


import utils.utils as utils
from utils.constants import *


class GenerationMode(enum.Enum):
    SINGLE_IMAGE = 0,
    INTERPOLATION = 1,
    VECTOR_ARITHMETIC = 2


def postprocess_generated_img(generated_img_tensor):
    assert isinstance(generated_img_tensor, torch.Tensor), f'Expected PyTorch tensor but got {type(generated_img_tensor)}.'

    # Move the tensor from GPU to CPU, convert to numpy array, extract 0th batch, move the image channel
    # from 0th to 2nd position (CHW -> HWC)
    generated_img = np.moveaxis(generated_img_tensor.to('cpu').numpy()[0], 0, 2)

    # If grayscale image repeat 3 times to get RGB image (for generators trained on MNIST)
    if generated_img.shape[2] == 1:
        generated_img = np.repeat(generated_img,  3, axis=2)

    # Imagery is in the range [-1, 1] (generator has tanh as the output activation) move it into [0, 1] range
    generated_img -= np.min(generated_img)
    generated_img /= np.max(generated_img)

    return generated_img


def generate_from_random_latent_vector(generator, cgan_digit=None):
    with torch.no_grad():
        latent_vector = utils.get_gaussian_latent_batch(1, next(generator.parameters()).device)

        if cgan_digit is None:
            generated_img = postprocess_generated_img(generator(latent_vector))
        else:  # condition and generate the digit specified by cgan_digit
            ref_label = torch.tensor([cgan_digit], dtype=torch.int64)
            ref_label_one_hot_encoding = torch.nn.functional.one_hot(ref_label, MNIST_NUM_CLASSES).type(torch.FloatTensor).to(next(generator.parameters()).device)
            generated_img = postprocess_generated_img(generator(latent_vector, ref_label_one_hot_encoding))

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


def display_vector_arithmetic_results(imgs_to_display):
    fig = plt.figure(figsize=(6, 6))
    title_fontsize = 'x-small'
    num_display_imgs = 7
    titles = ['happy women', 'happy woman (avg)', 'neutral women', 'neutral woman (avg)', 'neutral men', 'neutral man (avg)', 'result - happy man']
    ax = np.zeros(num_display_imgs, dtype=object)
    assert len(imgs_to_display) == num_display_imgs, f'Expected {num_display_imgs} got {len(imgs_to_display)} images.'

    gs = fig.add_gridspec(5, 4, left=0.02, right=0.98, wspace=0.05, hspace=0.3)
    ax[0] = fig.add_subplot(gs[0, :3])
    ax[1] = fig.add_subplot(gs[0, 3])
    ax[2] = fig.add_subplot(gs[1, :3])
    ax[3] = fig.add_subplot(gs[1, 3])
    ax[4] = fig.add_subplot(gs[2, :3])
    ax[5] = fig.add_subplot(gs[2, 3])
    ax[6] = fig.add_subplot(gs[3:, 1:3])

    for i in range(num_display_imgs):
        ax[i].imshow(cv.resize(imgs_to_display[i], (0, 0), fx=3, fy=3, interpolation=cv.INTER_NEAREST))
        ax[i].set_title(titles[i], fontsize=title_fontsize)
        ax[i].tick_params(which='both', bottom=False, left=False, labelleft=False, labelbottom=False)

    plt.show()


def generate_new_images(model_name, cgan_digit=None, generation_mode=True, slerp=True, a=None, b=None, should_display=True):
    """ Generate imagery using pre-trained generator (using vanilla_generator_000000.pth by default)

    Args:
        model_name (str): model name you want to use (default lookup location is BINARIES_PATH).
        cgan_digit (int): if specified generate that exact digit.
        generation_mode (enum):  generate a single image from a random vector, interpolate between the 2 chosen latent
         vectors, or perform arithmetic over latent vectors (note: not every mode is supported for every model type)
        slerp (bool): if True use spherical interpolation otherwise use linear interpolation.
        a, b (numpy arrays): latent vectors, if set to None you'll be prompted to choose images you like,
         and use corresponding latent vectors instead.
        should_display (bool): Display the generated images before saving them.

    """

    model_path = os.path.join(BINARIES_PATH, model_name)
    assert os.path.exists(model_path), f'Could not find the model {model_path}. You first need to train your generator.'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the correct (vanilla, cGAN, DCGAN, ...) model, load the weights and put the model into evaluation mode
    model_state = torch.load(model_path)
    gan_type = model_state["gan_type"]
    print(f'Found {gan_type} GAN!')
    _, generator = utils.get_gan(device, gan_type)
    generator.load_state_dict(model_state["state_dict"], strict=True)
    generator.eval()

    # Generate a single image, save it and potentially display it
    if generation_mode == GenerationMode.SINGLE_IMAGE:
        generated_imgs_path = os.path.join(DATA_DIR_PATH, 'generated_imagery')
        os.makedirs(generated_imgs_path, exist_ok=True)

        generated_img, _ = generate_from_random_latent_vector(generator, cgan_digit if gan_type == GANType.CGAN.name else None)
        utils.save_and_maybe_display_image(generated_imgs_path, generated_img, should_display=should_display)

    # Pick 2 images you like between which you'd like to interpolate (by typing 'y' into console)
    elif generation_mode == GenerationMode.INTERPOLATION:
        assert gan_type == GANType.VANILLA.name or gan_type ==GANType.DCGAN.name, f'Got {gan_type} but only VANILLA/DCGAN are supported for the interpolation mode.'

        interpolation_name = "spherical" if slerp else "linear"
        interpolation_fn = spherical_interpolation if slerp else linear_interpolation

        grid_interpolated_imgs_path = os.path.join(DATA_DIR_PATH, 'interpolated_imagery')  # combined results dir
        decomposed_interpolated_imgs_path = os.path.join(grid_interpolated_imgs_path, f'tmp_{gan_type}_{interpolation_name}_dump')  # dump separate results
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
            print('Skipping latent vectors selection section and using cached ones.')
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
            generated_imgs.append(torch.tensor(np.moveaxis(generated_img, 2, 0)))

        interpolated_block_img = torch.stack(generated_imgs)
        interpolated_block_img = nn.Upsample(scale_factor=2.5, mode='nearest')(interpolated_block_img)
        save_image(interpolated_block_img, os.path.join(grid_interpolated_imgs_path, utils.get_available_file_name(grid_interpolated_imgs_path)), nrow=int(np.sqrt(num_interpolated_imgs)))

    elif generation_mode == GenerationMode.VECTOR_ARITHMETIC:
        assert gan_type == GANType.DCGAN.name, f'Got {gan_type} but only DCGAN is supported for arithmetic mode.'

        # Generate num_options face images and create a grid image from them
        num_options = 100
        generated_imgs = []
        latent_vectors = []
        padding = 2
        for i in range(num_options):
            generated_img, latent_vector = generate_from_random_latent_vector(generator)
            generated_imgs.append(torch.tensor(np.moveaxis(generated_img, 2, 0)))  # make_grid expects CHW format
            latent_vectors.append(latent_vector)
        stacked_tensor_imgs = torch.stack(generated_imgs)
        final_tensor_img = make_grid(stacked_tensor_imgs, nrow=int(np.sqrt(num_options)), padding=padding)
        display_img = np.moveaxis(final_tensor_img.numpy(), 0, 2)

        # For storing latent vectors
        num_of_vectors_per_category = 3
        happy_woman_latent_vectors = []
        neutral_woman_latent_vectors = []
        neutral_man_latent_vectors = []

        # Make it easy - by clicking on the plot you pick the image.
        def onclick(event):
            if event.dblclick:
                pass
            else:  # single click
                if event.button == 1:  # left click
                    x_coord = event.xdata
                    y_coord = event.ydata
                    column = int(x_coord / (64 + padding))
                    row = int(y_coord / (64 + padding))

                    # Store latent vector corresponding to the image that the user clicked on.
                    if len(happy_woman_latent_vectors) < num_of_vectors_per_category:
                        happy_woman_latent_vectors.append(latent_vectors[10*row + column])
                        print(f'Picked image row={row}, column={column} as {len(happy_woman_latent_vectors)}. happy woman.')
                    elif len(neutral_woman_latent_vectors) < num_of_vectors_per_category:
                        neutral_woman_latent_vectors.append(latent_vectors[10*row + column])
                        print(f'Picked image row={row}, column={column} as {len(neutral_woman_latent_vectors)}. neutral woman.')
                    elif len(neutral_man_latent_vectors) < num_of_vectors_per_category:
                        neutral_man_latent_vectors.append(latent_vectors[10*row + column])
                        print(f'Picked image row={row}, column={column} as {len(neutral_man_latent_vectors)}. neutral man.')
                    else:
                        plt.close()

        plt.figure(figsize=(10, 10))
        plt.imshow(display_img)
        # This is just an example you could also pick 3 neutral woman images with sunglasses, etc.
        plt.title('Click on 3 happy women, 3 neutral women and \n 3 neutral men images (order matters!)')
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        plt.gcf().canvas.mpl_disconnect(cid)
        print('Done choosing images.')

        # Calculate the average latent vector for every category (happy woman, neutral woman, neutral man)
        happy_woman_avg_latent_vector = np.mean(np.array(happy_woman_latent_vectors), axis=0)
        neutral_woman_avg_latent_vector = np.mean(np.array(neutral_woman_latent_vectors), axis=0)
        neutral_man_avg_latent_vector = np.mean(np.array(neutral_man_latent_vectors), axis=0)

        # By subtracting neutral woman from the happy woman we capture the "vector of smiling". Adding that vector
        # to a neutral man we get a happy man's latent vector! Our latent space has amazingly beautiful structure!
        happy_man_latent_vector = neutral_man_avg_latent_vector + (happy_woman_avg_latent_vector - neutral_woman_avg_latent_vector)

        # Generate images from these latent vectors
        happy_women_imgs = np.hstack([generate_from_specified_numpy_latent_vector(generator, v) for v in happy_woman_latent_vectors])
        neutral_women_imgs = np.hstack([generate_from_specified_numpy_latent_vector(generator, v) for v in neutral_woman_latent_vectors])
        neutral_men_imgs = np.hstack([generate_from_specified_numpy_latent_vector(generator, v) for v in neutral_man_latent_vectors])

        happy_woman_avg_img = generate_from_specified_numpy_latent_vector(generator, happy_woman_avg_latent_vector)
        neutral_woman_avg_img = generate_from_specified_numpy_latent_vector(generator, neutral_woman_avg_latent_vector)
        neutral_man_avg_img = generate_from_specified_numpy_latent_vector(generator, neutral_man_avg_latent_vector)

        happy_man_img = generate_from_specified_numpy_latent_vector(generator, happy_man_latent_vector)

        display_vector_arithmetic_results([happy_women_imgs, happy_woman_avg_img, neutral_women_imgs, neutral_woman_avg_img, neutral_men_imgs, neutral_man_avg_img, happy_man_img])
    else:
        raise Exception(f'Generation mode not yet supported.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Pre-trained generator model name", default=r'VANILLA_000000.pth')
    parser.add_argument("--cgan_digit", type=int, help="Used only for cGAN - generate specified digit", default=3)
    parser.add_argument("--generation_mode", type=bool, help="Pick between 3 generation modes", default=GenerationMode.SINGLE_IMAGE)
    parser.add_argument("--slerp", type=bool, help="Should use spherical interpolation (default No)", default=False)
    parser.add_argument("--should_display", type=bool, help="Display intermediate results", default=True)
    args = parser.parse_args()

    # The first time you start generation in the interpolation mode it will cache a and b
    # which you'll choose the first time you run the it.
    a_path = os.path.join(DATA_DIR_PATH, 'interpolated_imagery', 'a.npy')
    b_path = os.path.join(DATA_DIR_PATH, 'interpolated_imagery', 'b.npy')
    latent_vector_a = np.load(a_path) if os.path.exists(a_path) else None
    latent_vector_b = np.load(b_path) if os.path.exists(b_path) else None

    generate_new_images(
        args.model_name,
        args.cgan_digit,
        generation_mode=args.generation_mode,
        slerp=args.slerp,
        a=latent_vector_a,
        b=latent_vector_b,
        should_display=args.should_display)
