import os

from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import numpy as np
from keras.preprocessing import image

from src import Config

face_size = 224

def predictVGG(face_images, model):
    model = VGGFace(model=model, include_top=False,
                    input_shape=(face_size, face_size, 3))
    face_image_array = np.stack(face_images, axis=0)
    embeddings = model.predict(face_image_array)
    return np.squeeze(embeddings)  # remove unnecessary dimensions


def get_vgg_embeddings(face_paths, model):
    preprocessed_faces, unprocessed_faces = get_faces(face_paths)
    # VGG
    vgg_embeddings = predictVGG(preprocessed_faces, model)  # returns embeddings as ndarray of shape (imageCount, embeddingSize)
    return vgg_embeddings

def get_vgg_embeddings_and_faces(face_paths, model):
    preprocessed_faces, unprocessed_faces = get_faces(face_paths)
    # VGG
    vgg_embeddings = predictVGG(preprocessed_faces, model)  # returns embeddings as ndarray of shape (imageCount, embeddingSize)
    return vgg_embeddings, unprocessed_faces


# %%

def get_faces(face_paths):
    processed_faces = []
    unprocessed_faces = []
    for idx, face_path in enumerate(face_paths):
        # read image
        face = image.load_img(face_path, target_size=(face_size, face_size))
        face = image.img_to_array(face)
        unprocessed_faces.append(face)
        face = np.expand_dims(face, axis=0)
        face = utils.preprocess_input(face, version=2)  # or version=2
        processed_faces.append(np.squeeze(face))
    return processed_faces, unprocessed_faces

def get_embeddings_and_paths(folder_path):
    face_paths = []
    folder_path = os.path.join(folder_path, Config.extracted_faces_path)
    for in_filename in os.listdir(folder_path):
        if not in_filename.endswith('.png'): continue
        face_paths.append(os.path.join(folder_path, in_filename))
    vgg_embeddings, face_imgs = get_vgg_embeddings_and_faces(face_paths, 'senet50')
    return vgg_embeddings, face_paths
