import argparse
import cv2
import os
import configparser
import sys
import csv
import numpy as np
import src.FotoExtractor.imextract.compare as cp
import src.FotoExtractor.imextract.backgroundremover as br
import src.FotoExtractor.imextract.rectextract as re
import src.FotoExtractor.imextract.facedetection as fd


def compare_images(img, path):
    """
    This compares the two images by the size, the differences in features and SSIM.

    :param img: ndarray - The image to compare to the ground truth.
    :param path: string - The path to the ground truth image to compare to.
    """
    truth = cv2.imread(path, cv2.IMREAD_COLOR)
    print("Difference percentage of the size: {} %".format(cp.compare_size(truth, img)))
    print("Difference percentage of the features: {} %".format(cp.compare_feature(truth, img, 20, 31.0)))
    print("SSIM: {} %".format(cp.ssim(truth, img)))


def get_frame_extracted_image(img):
    """
    This extracts the image from a surrounding frame if a frame is existing.

    :param img: ndarray - The image to extract from frame.
    :param config: dictionary - The configuration of the config file.
    :return: Extracted image.
    :rtype ndarray.
    """

    max_window_size = 0.1 # config.getfloat('ImageExtraction', 'MaxWindowSize')
    steps = 25 # config.getint('ImageExtraction', 'Steps')
    offset = 4 # config.getint('ImageExtraction', 'Offset')
    if not max_window_size or not steps or not offset:
        sys.exit("Error in the config file!")

    img = re.remove_border(img, steps, max_window_size, offset)

    return img


def get_background_extracted_images(img):
    """
    Approximately cuts out all the photos in the input image.

    :param img: the scanned site of an album.
    :param config: dictionary - The configuration of the config file.
    :returns: array of cut out images.
    :rtype: array
    """
    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)

    background_color = br.get_primary_background_color(img)

    spot_size = 200 # config.getint('BackgroundRemover', 'SpotSize')
    if not spot_size:
        sys.exit("Error in the config file!")

    background_location = br.get_background_spot(img, background_color, spot_size)

    binary_threshold = 25 # config.getint('BackgroundRemover', 'BinaryThreshold')
    if not binary_threshold:
        sys.exit("Error in the config file!")

    binary_img = br.generate_binary_background_image(img, background_color, binary_threshold)
    binary_background_img = br.separate_background(binary_img, background_location)

    min_area = -100 # config.getint('BackgroundRemover', 'MinImageSize')
    max_dimension_relation = 2.5 # config.getfloat('BackgroundRemover', 'MaxRelationImageDimensions')
    image_padding = 10 # config.getint('BackgroundRemover', 'ImagePadding')
    if not min_area or not max_dimension_relation or not image_padding:
        sys.exit("Error in the config file!")

    cropped_images = br.crop_image_rectangles(img, binary_background_img)

    feature_threshold = 10 # config.getint('BackgroundRemover', 'FeatureThreshold')
    if not feature_threshold:
        sys.exit("Error in the config file!")

    valid_cropped_images = br.validate_cropped_images(cropped_images, feature_threshold)

    return valid_cropped_images


def get_detected_faces(img, frontal_classifier, profile_classifier, out_path, name):
    """
    Detects the faces in the cut out images, marks them in the resulting image with a green rectangle and saves them seperately.

    :param img: ndarray - The image to detect faces.
    :param frontal_classifier: The Casscade Classifier to detect frontal faces.
    :param profile_classifier: The Casscade Classifier to detect faces in profile.
    :param path: Path to where the detected faces should be stored.
    :param name: Notation of the current image.
    :return:
    """

    scale = 1.2 # config.getfloat('FaceDetection', 'ScaleFactor')
    neighbors = 5 # config.getint('FaceDetection', 'Neighbors')

    faces_list = fd.detect_faces(img, frontal_classifier, profile_classifier, scale, neighbors)

    if faces_list:

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        j = 0
        for (x, y, w, h) in faces_list:
            sub_face = img[y:y+h, x:x+w]
            cv2.imwrite(out_path + name + '_' + str(j) + ".png", sub_face)
            j += 1



def main(args, config):
    """
    Main function to run the program with the given arguments.

    :param args: dictionary - The Arguments given by the start of the programm.
    :param config: dictionary - The configuration of the config file.
    """
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)

    if img is None:
        sys.exit("Error no image found!")

    if args.compare:
        compare_images(img, args.compare)
    else:
        full_name = os.path.basename(args.image)
        name = os.path.splitext(full_name)[0]

        cropped_images = get_background_extracted_images(img)

        for i, image in enumerate(cropped_images):

            img = get_frame_extracted_image(image)

            if args.face:

                path_frontal_classifier = config.get('FaceDetection', 'CascadePathFrontal')
                path_profile_classifier = config.get('FaceDetection', 'CascadePathProfile')

                if not path_frontal_classifier or not path_profile_classifier:
                    sys.exit("Error in the config file!")

                frontal_classifier = cv2.CascadeClassifier(path_frontal_classifier)
                profile_classifier = cv2.CascadeClassifier(path_profile_classifier)

                get_detected_faces(img, frontal_classifier, profile_classifier, config, args.result, name + '_' + str(i))

            if not os.path.exists(args.result):
                os.makedirs(args.result)

            cv2.imwrite(args.result + '/' + name + '_' + str(i) + '.png', img)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image",
                        help="path to the image of the photo album page",
                        type=str,
                        default='input/example_data/01.tif'
                        )

    parser.add_argument("--result",
                        help="path to the directory to store the results",
                        type=str,
                        default="./output"
                        )

    parser.add_argument("--compare",
                        help="path to the image to compare the given image to",
                        type=str
                        )

    parser.add_argument("--config",
                        help="path to the config file",
                        type=str,
                        default="./config"
                        )

    parser.add_argument("--face",
                        help="enables face detection",
                        action="store_true"
                        )
    return parser.parse_args()

def extract_images(image_path, out_path):
    """
    Extracts photos from an album picture into the output path.
    :param image_path:
    :param out_path:
    :return:
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        sys.exit("Error no image found!")

    full_name = os.path.basename(image_path)
    name = os.path.splitext(full_name)[0]

    cropped_images = get_background_extracted_images(img)

    for i, image in enumerate(cropped_images):

        img = get_frame_extracted_image(image)
        if not os.path.exists(out_path):
            os.makedirs(out_path.result)

        cv2.imwrite(out_path + '/' + name + '_' + str(i) + '.png', img)


if __name__ == "__main__":
    arguments = parse_arguments()
    configuration = configparser.ConfigParser()
    configuration.read(arguments.config)
    main(arguments, configuration)
