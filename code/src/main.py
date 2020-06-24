import os
from src import Config
from src.FaceExtractor.extract_faces import extract_faces
from src.FotoExtractor.extract_images import extract_images, ExtractionException


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


if __name__ == '__main__':
    extrac_all_faces_from_all_albums()
