import os


import cv2 as cv
import numpy as np
import imageio


from .utils import load_image


def create_gif(frames_dir, out_path, downsample=1, img_width=None):
    assert os.path.splitext(out_path)[1].lower() == '.gif', f'Expected gif got {os.path.splitext(out_path)[1]}.'

    frame_paths = [os.path.join(frames_dir, frame_name) for cnt, frame_name in enumerate(os.listdir(frames_dir)) if frame_name.endswith('.jpg') and cnt % downsample == 0]

    if img_width is not None:  # overwrites over the old frames
        for frame_path in frame_paths:
            img = load_image(frame_path, target_shape=img_width)
            cv.imwrite(frame_path, np.uint8(img[:, :, ::-1] * 255))

    images = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimwrite(out_path, images, fps=5)
    print(f'Saved gif to {out_path}.')