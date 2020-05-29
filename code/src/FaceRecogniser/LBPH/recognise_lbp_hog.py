import os
from src import Config

import cv2
from cv2.face import LBPHFaceRecognizer_create

def predict_lbph(img_filename, base_path = Config.EXTRACTED_FACES_PATH):
    rec = LBPHFaceRecognizer_create()
    training_filepath = os.path.join('dat', 'opencv_lbphfaces.yml')
    if not os.path.isfile(training_filepath):
        raise FileNotFoundError('Training file not found!')
    rec.read(training_filepath)
    photo_filepath = os.path.join(base_path, img_filename)
    img=cv2.imread(photo_filepath, cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    id,conf=rec.predict(gray)
    return id, conf


if __name__=='__main__':
    for img_filename in os.listdir(Config.EXTRACTED_FACES_PATH):
        id, conf = predict_lbph(img_filename)
        print(f'{img_filename}: {id} ({conf})')