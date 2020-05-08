import os
from src import Config

import cv2
from cv2.face import LBPHFaceRecognizer_create

def predict(img_filename='02_4_1.png'):
    rec=LBPHFaceRecognizer_create()
    rec.read("training.yml")
    photo_filepath = os.path.join(Config.EXTRACTED_FACES_PATH, img_filename)
    img=cv2.imread(photo_filepath, cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    id,conf=rec.predict(gray)
    return id, conf


if __name__=='__main__':
    for img_filename in os.listdir(Config.EXTRACTED_FACES_PATH):
        id, conf = predict(img_filename)
        print(f'{img_filename}: {id} ({conf})')
