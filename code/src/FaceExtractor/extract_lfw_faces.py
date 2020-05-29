from src.FaceExtractor.extract_faces import extract_faces
import os
from src import Config

def extract_lfw_faces(filepath):
    """
    Extracting with the general extract method,
    but keeping the convention that extracted faces go into a subfolder in lfw.
    :param filepath:
    :return:
    """
    out_path = os.path.join(*os.path.split(filepath)[:-1], 'extracted_faces')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    extract_faces(filepath, out_path)


if __name__=='__main__':
    N = len(os.listdir(Config.LFW_PATH))
    for i, folder_name in enumerate(os.listdir(Config.LFW_PATH)):
        for file_name in os.listdir(os.path.join(Config.LFW_PATH, folder_name)):
            file_path = os.path.join(Config.LFW_PATH, folder_name, file_name)
            if not os.path.isfile(file_path):
                continue
            try:
                extract_lfw_faces(file_path)
            except:
                raise Exception(f'Face extraction failed for {file_name}')
        print(f'{i}/{N}')