# https://github.com/AsankaD7/Face-Recognition-Train-YML-Python/blob/master/trainer.py
import os
from cv2.face import LBPHFaceRecognizer_create
import cv2
import numpy as np
from PIL import Image
from src import  Config


def getImages(training_filenames):
    # training_filenames = ['02_3_1.png', '02_1_0.png', '02_0_2.png', '01_3_0.png', '01_4_0.png', '02_4_0.png']
    # 02_4_1.png is also a positive, but used for testing
    path = Config.EXTRACTED_FACES_PATH
    imagePaths=[os.path.join(path,f) for f in training_filenames]
    faces=[]
    for i, imagePath in enumerate(imagePaths):
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        faces.append(faceNp)
    return faces

if __name__=='__main__':
    recognizer = LBPHFaceRecognizer_create()
    positive_filenames = ['02_3_1.png', '02_1_0.png', '02_0_2.png', '02_4_1.png']
    training_filenames = os.listdir(Config.EXTRACTED_FACES_PATH)
    for f in os.listdir(Config.EXTRACTED_FACES_PATH)[:10]:
        training_filenames.remove(f)
    faces=getImages(training_filenames)
    training_labels = [1 if file in positive_filenames else 0 for file in training_filenames]
    recognizer.train(faces,np.array(training_labels))
    recognizer.save('training.yml')
