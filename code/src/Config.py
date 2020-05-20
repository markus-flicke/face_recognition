import os
import logging
# Face Extractor
import sys

FACE_FRONTAL_EXTRACTOR_CLASSIFIER_PATH = os.path.join('src', 'FaceExtractor', 'classifiers',
                                                      'haarcascade_frontalface.xml')
FACE_PROFILE_EXTRACTOR_CLASSIFIER_PATH = os.path.join('src', 'FaceExtractor', 'classifiers',
                                                      'haarcascade_profileface.xml')
ANDREAS_ALBUMS_PATH = os.path.join('dat', 'AndreasAlbums')
ALBUM2_PATH = os.path.join(ANDREAS_ALBUMS_PATH, 'A2')

EXTRACTED_FACES_PATH = os.path.join(ANDREAS_ALBUMS_PATH, 'extracted_faces')
EXTRACTED_PHOTOS_PATH = os.path.join(ANDREAS_ALBUMS_PATH, 'extracted_photos')

LFW_PATH = os.path.join('dat', 'lfw')


# Logging configuration setup
def setup_logging(logfile=os.path.join('log', 'face_recoginition.log')):
    LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOGGING_LEVEL = logging.DEBUG
    logging.basicConfig(filename=logfile,
                        filemode='w',
                        level=LOGGING_LEVEL,
                        format=LOGGING_FORMAT,
                        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOGGING_LEVEL)
    handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    logging.getLogger().addHandler(handler)


def create_directories():
    os.makedirs(ANDREAS_ALBUMS_PATH, exist_ok=True)
    os.makedirs(ALBUM2_PATH, exist_ok=True)
    os.makedirs(EXTRACTED_FACES_PATH, exist_ok=True)
    os.makedirs(EXTRACTED_PHOTOS_PATH, exist_ok=True)

create_directories()
