import cv2
import random
import os

import src.Config as Config
from src.DataLoader import DataLoader


def load_images():
    face_images = []
    faceSize = 64
    image_paths, _ = DataLoader().load_lfw_complete()
    number_of_images = len(image_paths)

    print('Load images')
    for idx, path in enumerate(image_paths):
        # read image
        face = cv2.imread(path)
        # Convert image to the same channel as Album-Dataset
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        # Resize image to a smaller size, similar to A1 and A2
        face_dim = (faceSize, faceSize)
        resized_face = cv2.resize(face, dsize=face_dim, interpolation=cv2.INTER_CUBIC)
        print('image {} of {}'.format(idx, number_of_images))

        face_images.append(resized_face)

    print('Loading images is finished')
    return face_images, image_paths


def perturb_images(images):
    perturbed_images = []
    kernel_sizes_gauss = range(1, 64, 2)
    kernel_sizes_median = range(1, 7, 2)

    print('Start perturbing images')
    for image in images:
        # Kernel size decides how strong blurring is.
        kernel_size_gauss = random.choice(kernel_sizes_gauss)
        gaussian = cv2.GaussianBlur(image, ksize=(kernel_size_gauss, kernel_size_gauss), sigmaX=1)
        kernel_size_median = random.choice(kernel_sizes_median)
        median = cv2.medianBlur(gaussian, ksize=kernel_size_median)

        perturbed_images.append(median)

    print('finished perturbing images')

    return perturbed_images


def modify_old_paths(paths):
    mod_paths = []

    for path in paths:
        mod_path = path.replace(Config.LFW_PATH, Config.LFW_LOW_RES_PATH)
        mod_paths.append(mod_path)

    return mod_paths


def save_images(images, old_paths):
    new_paths = modify_old_paths(old_paths)
    number_of_files = len(old_paths)
    idx = 0

    print('Save images')
    for path, image in zip(new_paths, images):
        split_path = path.split('extracted_faces')
        folder = split_path[0] + 'extracted_faces'
        if not os.path.exists(folder):
            os.makedirs(folder)
        print('image {} of {}'.format(idx, number_of_files))
        cv2.imwrite(path, image)
        idx += 1

    print('finished saving')


if __name__ == "__main__":
    face_images, orig_paths = load_images()
    perturbed_images = perturb_images(face_images)
    save_images(perturbed_images, orig_paths)
