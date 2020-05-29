from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torchvision
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from operator import itemgetter
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


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


dataset = datasets.ImageFolder('../../dat/LFW/test/')
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

dists = [[((e1 - e2).norm().item()) ** 2 for e2 in embeddings] for e1 in embeddings]
classes = np.sort(list(dict.fromkeys(names)))

top3_predictions = get_top3_predictions(dists, names)
top3_acc = top3_accuracy(top3_predictions)
print('top 3 accuracy: {:.2f}'.format(top3_acc))
