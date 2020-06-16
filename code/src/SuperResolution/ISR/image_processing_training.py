import cv2
import os
import numpy as np


def noisy(noise_typ, image):
    '''

    :param noise_typ:
        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
        'nothin'    do nothing
    :param image: ndarray image
    :return: noisy image
    '''
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + 10 * gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = (2 ** np.ceil(np.log2(vals)))
        noisy = np.random.poisson(0.5 * image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    elif noise_typ == "nothing":
        return image


face_dim_low = 64
face_dim_high = 128
noises = ['gauss', 'poisson', 'speckle', 's&p', 'nothing']

for root, dirs, files in os.walk("dat\\LFW\\all"):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)
        path_to_file = root + '\\' + file
        image = cv2.imread(path_to_file)

        noise = noises[np.random.randint(5)]

        file_path_high = 'dat\\LFW\\train\\high_res\\{}'.format(file)
        high_res = cv2.resize(image, dsize=(face_dim_high, face_dim_high), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(file_path_high, high_res)

        file_path_low = 'dat\\LFW\\train\\low_res\\{}'.format(file)
        low_res = noisy(noise, image)
        low_res = cv2.resize(low_res, dsize=(face_dim_low, face_dim_low), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(file_path_low, low_res)

for root, dirs, files in os.walk("dat\\LFW\\test"):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)
        path_to_file = root + '\\' + file
        image = cv2.imread(path_to_file)

        noise = noises[np.random.randint(5)]

        file_path_high = 'dat\\LFW\\valid\\high_res\\{}'.format(file)
        high_res = cv2.resize(image, dsize=(face_dim_high, face_dim_high), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(file_path_high, high_res)

        file_path_low = 'dat\\LFW\\valid\\low_res\\{}'.format(file)
        low_res = noisy(noise, image)
        low_res = cv2.resize(low_res, dsize=(face_dim_low, face_dim_low), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(file_path_low, low_res)
