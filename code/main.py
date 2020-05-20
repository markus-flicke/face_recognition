import os
from src import Config
from src.FaceExtractor.extract_faces import extract_faces
from src.FotoExtractor.extract_images import extract_images
from pathlib import Path


def extrac_all_faces_from_all_albums():
    """
    Extracts all photos and then all faces from all album pages in dat/album_pages
    Not sure if the numbering is deterministic. Better to not rerun this method to keep the dataset constant
    :return:
    """
    Config.setup_logging()
    # Extract images from album pages
    in_path = './dat/AndreasAlbums/A2'
    out_path = './dat/extracted_photos'

    for album_page in os.listdir(in_path):
        extract_images(in_path + '/' + album_page, out_path)

    # Extract faces from images
    for in_filename in os.listdir(Path(out_path)):
        if not in_filename.endswith('.png'): continue
        extract_faces(Path(out_path) / in_filename)


if __name__ == '__main__':
    faces = extrac_all_faces_from_all_albums()
print()
