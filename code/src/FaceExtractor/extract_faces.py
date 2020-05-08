import src.Config as Config
import cv2
import os
import logging
import src.FaceExtractor.facedetection as fd

class NoImageReadException(Exception):
    "If a file in the photos path cannot be read"


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

    scale = 1.2
    neighbors = 5

    faces_list = fd.detect_faces(img, frontal_classifier, profile_classifier, scale, neighbors)

    if faces_list:
        j = 0
        for (x, y, w, h) in faces_list:
            sub_face = img[y:y+h, x:x+w]
            filepath = os.path.join(out_path, name + '_' + str(j) + ".png")
            cv2.imwrite(filepath, sub_face)
            logging.debug(f'Face extracted: {filepath}')
            j += 1


def extract_faces(photo_filepath, out_dir=Config.EXTRACTED_FACES_PATH):
    """
    Crops all faces on a png photo and saves them to individual png files.
    :param photo_filepath:
    :param out_dir:
    :return:
    """
    # Skipping photos that have already been extracted
    in_filename = os.path.split(photo_filepath)[-1].split('.')[0]
    if f'{in_filename}_0.png' in os.listdir(out_dir):
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
                       os.path.join(out_dir, ''), os.path.split(photo_filepath)[-1].split('.')[-2])


if __name__ == '__main__':
    Config.setup_logging()
    for in_filename in os.listdir(Config.EXTRACTED_PHOTOS_PATH):
        if not in_filename.endswith('.png'): continue
        extract_faces(os.path.join(Config.EXTRACTED_PHOTOS_PATH, in_filename))
