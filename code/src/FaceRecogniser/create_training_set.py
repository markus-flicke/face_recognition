# https://github.com/AsankaD7/Face-Recognition-Train-YML-Python/blob/master/trainer.py
import os
from cv2.face import LBPHFaceRecognizer_create
import cv2
import numpy as np
from PIL import Image
from src import  Config


def getImagesWithID(path):
    training_filenames = ['02_3_1.png', '02_1_0.png', '02_0_2.png', '01_3_0.png', '01_4_0.png', '02_4_0.png']
    # 02_4_1.png is also a positive, but used for testing
    training_labels = [1,1,1,0,0,0]
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    imagePaths = [path for path in imagePaths if os.path.split(path)[-1] in training_filenames]
    faces=[]
    IDs=[]
    for i, imagePath in enumerate(imagePaths):
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=training_labels[i]
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return IDs,faces

if __name__=='__main__':
    recognizer = LBPHFaceRecognizer_create()
    path = Config.EXTRACTED_FACES_PATH
    Ids,faces=getImagesWithID(path)
    recognizer.train(faces,np.array(Ids))
    recognizer.save('training.yml')
    cv2.destroyAllWindows()