import os


import cv2 as cv
import numpy as np
import imageio


from .utils import load_image


def create_video_from_intermediate_results(out_path):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    freq = 4
    resolution = (256, 256)
    out_video = cv.VideoWriter(out_path, fourcc, freq, resolution)


def create_gif(frames_dir, out_path, img_width=None):
    assert os.path.splitext(out_path)[1].lower() == '.gif', f'Expected gif got {os.path.splitext(out_path)[1]}.'

    frame_paths = [os.path.join(frames_dir, frame_name) for frame_name in os.listdir(frames_dir) if frame_name.endswith('.jpg')]

    if img_width is not None:
        for frame_path in frame_paths:
            img = load_image(frame_path, target_shape=img_width)
            cv.imwrite(frame_path, np.uint8(img[:, :, ::-1] * 255))

    images = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimwrite(out_path, images, fps=30)
    print(f'Saved gif to {out_path}.')