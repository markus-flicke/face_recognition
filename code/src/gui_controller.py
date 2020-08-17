import os
from src import Config
from src.FaceExtractor.extract_faces import extract_faces
from src.FaceRecogniser.VGG2.vgg2 import get_vgg_embeddings_and_faces
from src.FaceRecogniser.clustering import cluster_predictions, get_closest_clusters
from src.FotoExtractor.extract_images import extract_images, ExtractionException
import numpy as np


def get_clusters(embeddings, threshold):
    threshold = threshold / 100
    predictions = cluster_predictions(embeddings, threshold, 2, 'cosine')
    predictions_clean = predictions[predictions != -1]
    cluster_count = np.unique(predictions_clean).shape[0]
    avg_imgs_per_cluster = predictions_clean.shape[0] / cluster_count
    closest_clusters = get_closest_clusters(embeddings, predictions_clean)
    return predictions, cluster_count, avg_imgs_per_cluster, closest_clusters


def extract_faces_from_folder(image_folder, isAlbum=True):
    extracted_faces_path = os.path.join(image_folder, Config.extracted_faces_path)
    if (isAlbum):
        extracted_photos_path = os.path.join(image_folder, Config.extracted_photos_path)
        os.makedirs(extracted_photos_path, exist_ok=True)
        for album_page in os.listdir(image_folder):
            if not album_page.endswith('.tif'): continue
            try:
                extract_images(os.path.join(image_folder, album_page), extracted_photos_path)
            except ExtractionException:
                continue
    else:
        # if it is not an album but images the photos are already extracted
        extracted_photos_path = image_folder

    os.makedirs(extracted_faces_path, exist_ok=True)
    # Extract faces from images
    for in_filename in os.listdir(extracted_photos_path):
        if not in_filename.endswith('.png'): continue
        extract_faces(os.path.join(extracted_photos_path, in_filename),
                      extracted_faces_path)


def get_embeddings_and_imgs(folder_path):
    face_paths = []
    folder_path = os.path.join(folder_path, Config.extracted_faces_path)
    for in_filename in os.listdir(folder_path):
        if not in_filename.endswith('.png'): continue
        face_paths.append(os.path.join(folder_path, in_filename))
    vgg_embeddings, face_imgs = get_vgg_embeddings_and_faces(face_paths, 'senet50')
    return vgg_embeddings, face_paths

