import numpy as np
import cv2

from DataLoader import *
from ISR.models import RDN, RRDN
from PIL import Image
import yaml
import ISR.utils.metrics


def load_images():
    face_paths, labels = DataLoader().load_A2()
    faceSize = 64
    face_list = []
    file_names = []

    # prepare faces for prediction
    for idx, face_path in enumerate(face_paths):
        # read image
        face = cv2.imread(face_path)
        file_name = face_path.split('\\')
        file_name = file_name[-1]
        # resize Image
        face_dim = (faceSize, faceSize)
        resized_face = cv2.resize(face, dsize=face_dim, interpolation=cv2.INTER_CUBIC)
        face_list.append(resized_face)
        file_names.append(file_name)

    return face_list, file_names


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


with open('./weights/rrdn-C4-D3-G64-G064-T10-x2/2020-06-16_0750/session_config.yml', 'r') as f:
    print(yaml.load(f))

with open('./weights/rrdn-C4-D3-G64-G064-T10-x2/2020-06-16_0750/session_config.yml', 'r') as f:
    test = yaml.load(f)

gen_test = test['2020-06-16_0750']['training_parameters']['metrics']['generator']
# Load face images
images, file_names = load_images()
# Load RDN model
# rdn = RDN(weights='psnr-large')
# rrdn = RRDN(weights='gans')
rrdn = RRDN(weights='/weights/rrdn-C4-D3-G64-G064-T10-x2/2020-06-16_0750/session_config.yml')

# Make predictions
for image, file_name in zip(images, file_names):
    # sr_img = rdn.predict(image)
    sr_img = rrdn.predict(image)
    save_image(sr_img, 'test_sr/{}'.format(file_name))
