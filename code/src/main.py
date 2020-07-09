import os
from src import Config
from src.DataLoader import DataLoader
from src.FaceExtractor.extract_faces import extract_faces
from src.FaceRecogniser.VGG2.vgg2 import get_vgg_embeddings
from src.FaceRecogniser.clustering import evaluate_best_threshold, cluster_predictions
from src.FotoExtractor.extract_images import extract_images, ExtractionException
import numpy as np
from pathlib import Path


def extrac_all_faces_from_all_albums():
    """
    Extracts all photos and then all faces from all album pages in dat/album_pages
    Not sure if the numbering is deterministic. Better to not rerun this method to keep the dataset constant
    :return:
    """
    Config.setup_logging()
    # Extract images from album pages
    in_path = Config.A1_PATH
    out_path = Config.A1_EXTRACTED_PHOTOS_PATH
    for album_page in os.listdir(in_path):
        try:
            extract_images(os.path.join(in_path, album_page), out_path)
        except ExtractionException:
            continue
    # Extract faces from images
    for in_filename in os.listdir(Config.A1_EXTRACTED_PHOTOS_PATH):
        if not in_filename.endswith('.png'): continue
        extract_faces(os.path.join(Config.A1_EXTRACTED_PHOTOS_PATH, in_filename))


def get_clusters(embeddings, threshold):
    threshold = threshold / 100
    predictions = cluster_predictions(embeddings, threshold, 2, 'cosine')
    predictions_clean = predictions[predictions != -1]
    cluster_count = np.unique(predictions_clean).shape[0]
    avg_imgs_per_cluster = predictions_clean.shape[0] / cluster_count
    return predictions, cluster_count, avg_imgs_per_cluster


def extract_faces_from_folder(image_folder, isAlbum=True):
    extracted_faces_path = os.path.join(image_folder, Config.extracted_faces_path)
    if (isAlbum):
        extracted_photos_path = os.path.join(image_folder, Config.extracted_photos_path)
        os.makedirs(extracted_photos_path, exist_ok=True)
        for album_page in os.listdir(image_folder):
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


def get_embeddings(folder_path):
    face_paths = []
    folder_path = os.path.join(folder_path, Config.extracted_faces_path)
    for in_filename in os.listdir(folder_path):
        if not in_filename.endswith('.png'): continue
        face_paths.append(os.path.join(folder_path, in_filename))
    face_imgs, vgg_embeddings = get_vgg_embeddings(face_paths, 'resnet50')
    return face_imgs, vgg_embeddings


if __name__ == '__main__':
    extrac_all_faces_from_all_albums()
