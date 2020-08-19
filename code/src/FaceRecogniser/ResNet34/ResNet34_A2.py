import numpy as np
import face_recognition


def get_ResNet_embeddings(filepaths, labels):
    labels = np.array(labels)
    filepaths = np.array(filepaths)
    no_face_c = 0
    X = np.zeros((len(filepaths), 128))
    for i, filepath in enumerate(filepaths):
        img = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(img)
        # ToDo: Some cropped images won't look like a face to ReseNet-34
        if not encodings:
            no_face_c += 1
            continue

        X[i] = np.array(encodings)
    print(f'No encodings found for {no_face_c} faces.')
    return X[(X != 0).all(axis = 1)], labels[(X != 0).all(axis = 1)], filepaths[(X != 0).all(axis = 1)]
