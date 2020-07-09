from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import numpy as np
from keras.preprocessing import image

face_size = 224

def predictVGG(face_images, model):
    model = VGGFace(model=model, include_top=False,
                    input_shape=(face_size, face_size, 3))
    face_image_array = np.stack(face_images, axis=0)
    embeddings = model.predict(face_image_array)
    return np.squeeze(embeddings)  # remove unnecessary dimensions


def get_vgg_embeddings(face_paths, model):
    face_list = []
    face_imgs = []

    # prepare faces for prediction
    for idx, face_path in enumerate(face_paths):
        # read image
        face = image.load_img(face_path, target_size=(face_size, face_size))
        face = image.img_to_array(face)
        face_imgs.append(face)
        face = np.expand_dims(face, axis=0)
        face = utils.preprocess_input(face, version=2)  # or version=2
        # face = cv2.imread(face_path)
        # resize Image
        # face_dim = (face_size, face_size)
        # resized_face = cv2.resize(face, dsize=face_dim, interpolation=cv2.INTER_CUBIC)
        face_list.append(np.squeeze(face))

    # VGG
    vgg_embeddings = predictVGG(face_list, model)  # returns embeddings as ndarray of shape (imageCount, embeddingSize)

    return face_imgs, vgg_embeddings

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
