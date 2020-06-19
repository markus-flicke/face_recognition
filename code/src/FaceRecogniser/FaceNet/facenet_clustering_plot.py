from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torchvision
import csv
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

import numpy as np
import os

workers = 0 if os.name == 'nt' else 4
image_size = 128

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)


def collate_fn(x):
    return x[0]


dataset = datasets.ImageFolder('dat/AndreasAlbums/A2/extracted_faces'
                               , transform=torchvision.transforms.Resize((image_size, image_size))
                               )
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []

with open('dat/AndreasAlbums/A2/labels.csv') as csv_file:
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

dist_thresh = 0.3

db = DBSCAN(eps=0.001, min_samples=1)
db.fit(embeddings)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
labels_true = names

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = embeddings[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = embeddings[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.show()
