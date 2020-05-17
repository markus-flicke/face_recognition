# https://github.com/AsankaD7/Face-Recognition-Train-YML-Python/blob/master/trainer.py
import os
from cv2.face import LBPHFaceRecognizer_create

import numpy as np
from PIL import Image
from src import  Config


def _open_images(training_filenames, path):
    """
    Opens all images in trainig_filenames and returns them as a list.
    If you run into RAM problems because of this, turn it into a generator.
    :param training_filenames:
    :return:
    """
    imagePaths=[os.path.join(path,f) for f in training_filenames]
    faces=[]
    for i, imagePath in enumerate(imagePaths):
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        faces.append(faceNp)
    return faces

def train_opencv_classifier(training_filenames,
                            training_labels,
                            classifier = LBPHFaceRecognizer_create(),
                            training_path = Config.EXTRACTED_FACES_PATH):
    """
    Trains an OpenCV classifier and saves it to yml.
    :param training_filenames:
    :param training_labels:
    :return:
    """
    faces = _open_images(training_filenames, training_path)
    classifier.train(faces, np.array(training_labels))
    # The lbph face recogniser will be called 'opencv_lbphfaces'.
    classifier.save(os.path.join('dat', '{}.yml'.format(classifier.getDefaultName())))


if __name__=='__main__':
    recognizer = LBPHFaceRecognizer_create()
    positive_filenames = ['02_3_1.png', '02_1_0.png', '02_0_2.png', '02_4_1.png']
    training_filenames = os.listdir(Config.EXTRACTED_FACES_PATH)
    for f in os.listdir(Config.EXTRACTED_FACES_PATH)[:10]:
        training_filenames.remove(f)
    training_labels = [1 if file in positive_filenames else 0 for file in training_filenames]
    train_opencv_classifier(training_filenames, training_labels, LBPHFaceRecognizer_create())
