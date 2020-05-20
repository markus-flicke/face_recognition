import cv2
import os
import sys
import numpy as np
import src.FotoExtractor.backgroundremover as br
import src.FotoExtractor.rectextract as re


def get_frame_extracted_image(img):
    """
    This extracts the image from a surrounding frame if a frame is existing.

    :param img: ndarray - The image to extract from frame.
    :return: Extracted image.
    :rtype ndarray.
    """

    max_window_size = 0.1
    steps = 25
    offset = 4
    img = re.remove_border(img, steps, max_window_size, offset)
    return img


def get_background_extracted_images(img):
    """
    Approximately cuts out all the photos in the input image.

    :param img: the scanned site of an album.
    :returns: array of cut out images.
    :rtype: array
    """
    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)

    background_color = br.get_primary_background_color(img)
    spot_size = 200
    background_location = br.get_background_spot(img, background_color, spot_size)
    binary_threshold = 25
    binary_img = br.generate_binary_background_image(img, background_color, binary_threshold)
    binary_background_img = br.separate_background(binary_img, background_location)
    cropped_images = br.crop_image_rectangles(img, binary_background_img)
    feature_threshold = 10
    valid_cropped_images = br.validate_cropped_images(cropped_images, feature_threshold)
    return valid_cropped_images


class ExtractionException(Exception):
    """No Image found"""


def extract_images(image_path, out_path):
    """
    Extracts photos from an album picture into the output path.
    :param image_path:
    :param out_path:
    :return:
    """

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ExtractionException(image_path)

    full_name = os.path.basename(image_path)
    name = os.path.splitext(full_name)[0]

    cropped_images = get_background_extracted_images(img)

    for i, image in enumerate(cropped_images):

        img = get_frame_extracted_image(image)
        if not os.path.exists(out_path):
            os.makedirs(out_path.result)

        cv2.imwrite(out_path + '/' + name + '_' + str(i) + '.png', img)


if __name__ == '__main__':
    in_path = './dat/album_pages/'
    out_path = './dat/extracted_photos'
    for album_page in os.listdir(in_path):
        extract_images(in_path + album_page, out_path)
