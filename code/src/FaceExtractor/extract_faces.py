import src.Config as Config
import cv2
import os
from src.FotoExtractor.main import get_detected_faces
import logging


class NoImageReadException(Exception):
    "If a file in the photos path cannot be read"


def extract_faces(photo_filepath, out_path=Config.EXTRACTED_FACES_PATH):
    """
    Crops all faces on a png photo and saves them to individual png files.
    :param photo_filepath:
    :param out_path:
    :return:
    """
    # Skipping photos that have already been extracted
    in_filename = os.path.split(photo_filepath)[-1].split('.')[0]
    if f'{in_filename}_0.png' in os.listdir(out_path):
        logging.debug(f'No Face extraction necessary. Faces already extracted from photo: {in_filename}')
        return

    frontal_classifier = cv2.CascadeClassifier(Config.FACE_FRONTAL_EXTRACTOR_CLASSIFIER_PATH)
    profile_classifier = cv2.CascadeClassifier(Config.FACE_PROFILE_EXTRACTOR_CLASSIFIER_PATH)
    img = cv2.imread(photo_filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise NoImageReadException(
            f'Image reading failed for: {photo_filepath}')

    get_detected_faces(img,
                       frontal_classifier,
                       profile_classifier,
                       os.path.join(out_path, ''), photo_filepath.split('.')[-2])


if __name__ == '__main__':
    Config.setup_logging()
    for in_filename in os.listdir(Config.EXTRACTED_PHOTOS_PATH):
        if not in_filename.endswith('.png'): continue
        extract_faces(os.path.join(Config.EXTRACTED_PHOTOS_PATH, in_filename))
