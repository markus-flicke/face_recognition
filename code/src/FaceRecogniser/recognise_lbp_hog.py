import os
from src import Config

import cv2
from cv2.face import LBPHFaceRecognizer_create
from cv2.face import EigenFaceRecognizer_create

def predict(img_filename='02_4_1.png'):
    rec = LBPHFaceRecognizer_create()
    trining_filepath = os.path.join('dat', 'training.yml')
    if not os.path.isfile(trining_filepath):
        raise FileNotFoundError('Training file not found!')
    rec.read(trining_filepath)
    photo_filepath = os.path.join(Config.EXTRACTED_FACES_PATH, img_filename)
    img=cv2.imread(photo_filepath, cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    id,conf=rec.predict(gray)
    return id, conf


if __name__=='__main__':
    for img_filename in os.listdir(Config.EXTRACTED_FACES_PATH):
        id, conf = predict(img_filename)
        print(f'{img_filename}: {id} ({conf})')