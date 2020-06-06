import os

import main as m
from DataLoader import load_A2 as a2loader
from FaceRecogniser.VGG2 import vgg2
import Config
import cv2
from Config import face_image_size as faceSize
import FaceRecogniser.FaceNet.utils as utils
import numpy as np
import numpy.linalg

# extract faces and load face paths and label
# m.extrac_all_faces_from_all_albums()
face_paths, names = a2loader.load_A2()

face_list = []
# prepare faces for prediction
for idx, face_path in enumerate(face_paths):
    # read image
    face = cv2.imread(face_path)
    # resize Image
    face_dim = (faceSize, faceSize)
    resized_face = cv2.resize(face, dsize=face_dim, interpolation=cv2.INTER_CUBIC)
    face_list.append(resized_face)

# VGG
vgg_embeddings = vgg2.predictVGG(face_list) # returns embeddings as ndarray of shape (imageCount, embeddingSize)

dists = [[(np.linalg.norm(e1 - e2) ** 2) for e2 in vgg_embeddings] for e1 in vgg_embeddings]

classes = np.sort(list(dict.fromkeys(names)))

top3_predictions = utils.get_top3_predictions(dists, names)
top3_acc = utils.top3_accuracy(top3_predictions)
print('top 3 accuracy: {:.2f}'.format(top3_acc))

# bestimmen des Thresholds ab wann ein Bild als gleich erkannt wird
dist_thresh = 4000

# Bestimmen von Accuracy, Precision, Recall und F1 aus der Confusion-Matrix
tp, fp, fn, tn = utils.confusion_matrix(dist_thresh, vgg_embeddings, classes, names, dists)
accuracies, precicions, recalls = utils.classification_meassures(tp, fp, fn, tn, classes)

accuracy = np.nanmean(np.array(list(accuracies.values())))
precicion = np.nanmean(np.array(list(precicions.values())))
recall = np.nanmean(np.array(list(recalls.values())))
f1 = 2 * (precicion * recall) / (precicion + recall)

print('Accuracy {:.2f}'.format(accuracy))
print('Precision {:.2f}'.format(precicion))
print('Recall {:.2f}'.format(recall))
print('F1-Score {:.2f}'.format(f1))




