import sys

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torchvision
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from operator import itemgetter

from sklearn.cluster import DBSCAN
from imutils import build_montages
import cv2

import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4
image_size = 128

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)


def collate_fn(x):
    return x[0]


def get_top3_predictions(dist, name):
    top = []
    for row in dist:
        dist_class = zip(row, name)
        dist_class = sorted(dist_class, key=itemgetter(0))
        class_name = dist_class[0][1]
        top1 = dist_class[1][1]
        top2 = dist_class[2][1]
        top3 = dist_class[3][1]
        top.append((class_name, [top1, top2, top3]))

    return top


def top3_accuracy(top3):
    correct_predicts = 0
    num_of_predictions = len(top3)

    for el in top3:
        class_name = el[0]
        for c in el[1]:
            if class_name == c:
                correct_predicts += 1

    return correct_predicts / num_of_predictions


dataset = datasets.ImageFolder('../../dat/AndreasAlbums/extracted_faces/'
                               , transform=torchvision.transforms.Resize((image_size, image_size))
                               )
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []

with open('../../dat/AndreasAlbums/labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        names.append(row[1])

names = names[1:]
names = [int(x) for x in names]

for img, lbl in loader:
    aligned.append(torch.from_numpy(np.array(img)))

aligned = torch.stack(aligned).to(device=device, dtype=torch.float)
aligned = aligned.permute(0, 3, 1, 2)

embeddings = []
step_size = 10
for i in range(0, len(aligned), step_size):
    if (i + step_size) < len(aligned):
        batch_aligned = aligned[i:i + step_size]
    else:
        batch_aligned = aligned[i:]

    batch_embedding = resnet(batch_aligned).detach().cpu()
    embeddings.append(batch_embedding.numpy())

embeddings = np.concatenate(np.array(embeddings), 0)

dists = [[(np.linalg.norm(e1 - e2) ** 2) for e2 in embeddings] for e1 in embeddings]
classes = np.sort(list(dict.fromkeys(names)))

top3_predictions = get_top3_predictions(dists, names)
top3_acc = top3_accuracy(top3_predictions)
print('top 3 accuracy: {:.2f}'.format(top3_acc))

dist_thresh = 0.3


def confusion_matrix(threshold):
    tp = dict(zip(classes, np.zeros(len(classes))))
    tn = tp.copy()
    fp = tp.copy()
    fn = tp.copy()
    for row in range(len(embeddings)):
        for col in range(len(embeddings)):
            current_class = names[row]
            if dists[row][col] <= threshold:
                if current_class == names[col]:
                    tp[current_class] += 1
                else:
                    fp[current_class] += 1
            else:
                if current_class == names[col]:
                    fn[current_class] += 1
                else:
                    tn[current_class] += 1
    return tp, fp, fn, tn


def classification_meassures(tp, fp, fn, tn):
    accuracies = dict(zip(classes, np.zeros(len(classes))))
    precicions = accuracies.copy()
    recalls = accuracies.copy()
    for key in accuracies:
        accuracies[key] = (tp[key] + tn[key]) / (fp[key] + fn[key] + tp[key] + tn[key])
        precicions[key] = tp[key] / (tp[key] + fp[key])
        recalls[key] = tp[key] / (tp[key] + fn[key])

    return accuracies, precicions, recalls


def compute_tpr_fpr(thresh):
    tpr = dict(zip(classes, np.zeros(len(classes))))
    tprs = []
    fpr = tpr.copy()
    fprs = []

    for t in thresh:
        tp, fp, fn, tn = confusion_matrix(t)
        for key in tpr:
            tpr[key] = tp[key] / (tp[key] + fn[key])
            tprs.append(np.mean(np.array(list(tpr.values()))))
            fpr[key] = fp[key] / (fp[key] + tn[key])
            fprs.append(np.mean(np.array(list(fpr.values()))))

    return tprs, fprs


# ===============================================

tp, fp, fn, tn = confusion_matrix(dist_thresh)
accuracies, precicions, recalls = classification_meassures(tp, fp, fn, tn)

accuracy = np.mean(np.array(list(accuracies.values())))
precicion = np.mean(np.array(list(precicions.values())))
recall = np.mean(np.array(list(recalls.values())))

print('Accuracy {:.2f}'.format(accuracy))
print('Precision {:.2f}'.format(precicion))
print('Recall {:.2f}'.format(recall))

th = np.linspace(0, 1.5, 100)
tpr, fpr = compute_tpr_fpr(th)
plt.plot(fpr, tpr)
plt.show()

# ================================================


# clt = DBSCAN(eps=0.3, min_samples=1)
# clt.fit(embeddings)
# labelIDs = np.unique(clt.labels_)
# numUniqueFaces = len(np.where(labelIDs > -1)[0])
# print("[INFO] # unique faces: {}".format(numUniqueFaces))

# loop over the unique face integers
# for labelID in labelIDs:
# find all indexes into the `data` array that belong to the
# current label ID, then randomly sample a maximum of 25 indexes
# from the set
# print("[INFO] faces for face ID: {}".format(labelID))
# idxs = np.where(clt.labels_ == labelID)[0]
# idxs = np.random.choice(idxs, size=min(25, len(idxs)),
#                         replace=False)
# initialize the list of faces to include in the montage
# faces = []

# loop over the sampled indexes
# for i in idxs:
# load the input image and extract the face ROI
# image = cv2.imread('../../dat/AndreasAlbums/extracted_faces/01/01_0_0.png')
# force resize the face ROI to 96x96 and then add it to the
# faces montage list
# try:
#     face = cv2.resize(image, (96, 96))
#     faces.append(face)
# except Exception as e:
#     print(str(e))

# cv2.imshow('test', image)
# cv2.waitKey(0)

# # create a montage using 96x96 "tiles" with 5 rows and 5 columns
# montage = build_montages(faces, (96, 96), (5, 5))[0]
# print(montage)
#
# # show the output montage
# title = "Face ID #{}".format(labelID)
# title = "Unknown Faces" if labelID == -1 else title
# cv2.imshow(title, montage)
# cv2.waitKey(0)

# print(pd.DataFrame(dists, columns=names, index=names))
