from src.FaceRecogniser.FaceNet.utils import *
import torch
import cv2

import numpy as np
import os

workers = 0 if os.name == 'nt' else 4


def predictFaceNet(face_images, model_type):
    '''

    :param face_images: resized images with 128x128
    :param modelType: "vggface2" or "casia-webface"
    :return:
    '''

    embeddings = []
    step_size = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet = Model(model_type, device).resnet
    images = torch.FloatTensor(face_images)
    images = images.permute(0, 3, 1, 2)
    images = images.to(device=device, dtype=torch.float)

    for i in range(0, len(face_images), step_size):
        if (i + step_size) < len(face_images):
            batch = images[i:i + step_size]
        else:
            batch = images[i:]

        batch_embedding = resnet(batch).detach().cpu()
        embeddings.append(batch_embedding.numpy())

    embeddings = np.concatenate(np.array(embeddings), 0)

    return embeddings


def get_faceNet_embeddings(face_paths, model_type):
    faceSize = 128
    face_list = []
    # prepare faces for prediction
    for idx, face_path in enumerate(face_paths):
        # read image
        face = cv2.imread(face_path)
        # resize Image
        face_dim = (faceSize, faceSize)
        resized_face = cv2.resize(face, dsize=face_dim, interpolation=cv2.INTER_CUBIC)
        face_list.append(resized_face)

    # FaceNet
    # returns embeddings as ndarray of shape (imageCount, embeddingSize)
    faceNet_embeddings = predictFaceNet(face_list, model_type)

    return faceNet_embeddings
