import os

import cv2
from cv2.face import LBPHFaceRecognizer_create

rec=LBPHFaceRecognizer_create()
rec.read("training.yml")
photo_filepath = os.path.join('dat','extracted_faces', '02_4_1.png')
img=cv2.imread(photo_filepath, cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
id,conf=rec.predict(gray)
print(id)
