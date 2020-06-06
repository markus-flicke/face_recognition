import io
import IPython
import PIL
import pandas as pd
import os
import sklearn
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, recall_score
import numpy as np
import Config
import cv2


def predictVGG(face_images):
    model = VGGFace(model='resnet50', include_top=False, input_shape=(Config.face_image_size,Config.face_image_size,3))
    face_image_array = np.stack(face_images, axis=0)
    embeddings = model.predict(face_image_array)
    return np.squeeze(embeddings) # remove unnecessary dimensions


def get_vgg_embeddings(face_paths):
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
    vgg_embeddings = vgg2.predictVGG(face_list)  # returns embeddings as ndarray of shape (imageCount, embeddingSize)

    return vgg_embeddings

# %%



# %%
# threshold = 0.2
# clusters = cluster(embeddings.squeeze(), 0.29, 2, "cosine")
# for label in np.unique(clusters.labels_):
#     plt.figure(figsize=(20, 10))
#     columns = 5
#     face_indices = np.where(clusters.labels_ == label)[0]
#     print(str(label))
#     for i, face_idx in enumerate(face_indices):
#         if i < 5:
#             plt.subplot(min(5, len(face_indices)) / columns + 1, columns, i + 1)
#             plt.imshow(faces[face_idx, :, :, :].squeeze().astype(np.uint8))
#
#         # %%
#
# np.unique(clusters.labels_)
# cv2.imshow('image', faces[1, :, :, :])
#
# # %%
#
#
# accuracies = []
# recalls = []
# f1s = []
# for idx1, embedding1 in enumerate(embeddings):
#     label = labels_df.loc[labels_df['filename'] == imageNames[idx1]].iat[0, 1]
#     ground_truth = (labels_df['label'] == label) & (labels_df['filename'] != imageNames[idx1])
#     predictions = []
#     for idx2, embedding2 in enumerate(embeddings):
#         if idx1 != idx2 and cosine(embedding1, embedding2) < threshold:
#             predictions.append(True)
#         else:
#             predictions.append(False)
#     acc = accuracy_score(predictions, ground_truth)
#     accuracies.append(acc)
#     recall = recall_score(predictions, ground_truth)
#     recalls.append(recall)
#     f1 = sklearn.metrics.f1_score(predictions, ground_truth);
#     f1s.append(f1)
#
# overall_acc = sum(accuracies) / len(accuracies)
# overall_recall = sum(recalls) / len(recalls)
# overall_f1 = sum(f1s) / len(f1s)
# print('Accuracy: ' + str(overall_acc))
# print('Recall: ' + str(overall_recall))

# %%
