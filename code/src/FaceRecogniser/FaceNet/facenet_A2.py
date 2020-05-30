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

'''
Image size Angabe für Resize der ausgeschnittenen Bilder.
FaceNet wurde auf 128x128 Bilder trainiert und hat hier die beste Perfomance.
'''
image_size = 128

# CPU oder GPU Aswertung?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


class Model:
    ''' Für den Parameter model muss entweder der String "vggface2" oder "casia-webface" angegeben werden '''

    def __init__(self, model):
        self.resnet = InceptionResnetV1(pretrained=model).eval().to(device)


def collate_fn(x):
    return x[0]


'''
Berechnet die Top 3 predictions

:dist: List Distanzmatrix NxN
:name: List Label der Testdatensätze 1xN

:return: List of tupel (ground_truth_label, [top_3_label]) 1xN
'''


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


'''
Berechnet ob das korrekte Label in den Tupeln steckt und berechnet die accuracy

:top3: List of tupel (ground_truth_label, [top_3_label]) 1xN

:return: Scalar accuracy
'''


def top3_accuracy(top3):
    correct_predicts = 0
    num_of_predictions = len(top3)

    for el in top3:
        class_name = el[0]
        for c in el[1]:
            if class_name == c:
                correct_predicts += 1

    return correct_predicts / num_of_predictions


'''
Berechnet die Confusion matrix

:threshold: Scalar für den zu verwendeten Schwellenwert

:return: Dict {class: value} für tp, fp, tn, fn
'''


def confusion_matrix(threshold, embeddings):
    tp = dict(zip(classes, np.zeros(len(classes))))
    tn = tp.copy()
    fp = tp.copy()
    fn = tp.copy()
    for row in range(len(embeddings)):
        for col in range(len(embeddings)):
            current_class = names[row]
            if row != col:
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


'''
Berechnet Accuracy, Precicion und Recall für jede Klasse

:tp, fp, fn, tn: Dict {class: value} für tp, fp, tn, fn

:return: Dict {class: value} für Accuracy, Precicion und Recall 
'''


def classification_meassures(tp, fp, fn, tn):
    accuracies = dict(zip(classes, np.zeros(len(classes))))
    precicions = accuracies.copy()
    recalls = accuracies.copy()
    for key in accuracies:
        accuracies[key] = (tp[key] + tn[key]) / (fp[key] + fn[key] + tp[key] + tn[key])
        precicions[key] = tp[key] / (tp[key] + fp[key])
        recalls[key] = tp[key] / (tp[key] + fn[key])

    return accuracies, precicions, recalls


'''
Berechnen von tpr und fpr für die ROC-Kurve
'''


def compute_tpr_fpr(thresh):
    tpr = dict(zip(classes, np.zeros(len(classes))))
    tprs = []
    fpr = tpr.copy()
    fprs = []

    for t in thresh:
        tp, fp, fn, tn = confusion_matrix(t, embeddings)
        for key in tpr:
            tpr[key] = tp[key] / (tp[key] + fn[key])
            tprs.append(np.nanmean(np.array(list(tpr.values()))))
            fpr[key] = fp[key] / (fp[key] + tn[key])
            fprs.append(np.nanmean(np.array(list(fpr.values()))))

    return tprs, fprs


# Dantensatz (ausgeschnittene Gesichter) wird geladen und auf die richtige Größe gebracht
dataset = datasets.ImageFolder('../../dat/AndreasAlbums/extracted_faces/'
                               , transform=torchvision.transforms.Resize((image_size, image_size))
                               )
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []

# Lesen der Labels
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
# Initialisieren des Netzwerkes
# Für den Parameter model muss entweder der String "vggface2" oder "casia-webface" angegeben werden
resnet = Model('casia-webface').resnet

# Berechnen der Embeddings pro Batch. Batch-Größe wird über step_size angegben
for i in range(0, len(aligned), step_size):
    if (i + step_size) < len(aligned):
        batch_aligned = aligned[i:i + step_size]
    else:
        batch_aligned = aligned[i:]

    batch_embedding = resnet(batch_aligned).detach().cpu()
    embeddings.append(batch_embedding.numpy())

embeddings = np.concatenate(np.array(embeddings), 0)

# Berechnen der Distanzmatrix
dists = [[(np.linalg.norm(e1 - e2) ** 2) for e2 in embeddings] for e1 in embeddings]
classes = np.sort(list(dict.fromkeys(names)))

# Bestimmen der Top-3 Accuracy
top3_predictions = get_top3_predictions(dists, names)
top3_acc = top3_accuracy(top3_predictions)
print('top 3 accuracy: {:.2f}'.format(top3_acc))

# bestimmen des Thresholds ab wann ein Bild als gleich erkannt wird
dist_thresh = 0.01

# Bestimmen von Accuracy, Precision, Recall und F1 aus der Confusion-Matrix
tp, fp, fn, tn = confusion_matrix(dist_thresh, embeddings)
accuracies, precicions, recalls = classification_meassures(tp, fp, fn, tn)

accuracy = np.nanmean(np.array(list(accuracies.values())))
precicion = np.nanmean(np.array(list(precicions.values())))
recall = np.nanmean(np.array(list(recalls.values())))
f1 = 2 * (precicion * recall) / (precicion + recall)

print('Accuracy {:.2f}'.format(accuracy))
print('Precision {:.2f}'.format(precicion))
print('Recall {:.2f}'.format(recall))
print('F1-Score {:.2f}'.format(f1))

# Plotten der ROC-Kurve
th = np.linspace(0, 1.5, 100)
tpr, fpr = compute_tpr_fpr(th)
plt.plot(fpr, tpr)
plt.show()