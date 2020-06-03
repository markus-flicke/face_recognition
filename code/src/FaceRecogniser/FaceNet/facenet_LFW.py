from facenet_pytorch import MTCNN, InceptionResnetV1
from src.FaceRecogniser.FaceNet.utils import *
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os

from sklearn.cluster import DBSCAN
from sklearn import metrics

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Initialisieren des Netzwerkes
# Für den Parameter model muss entweder der String "vggface2" oder "casia-webface" angegeben werden
resnet = Model('casia-webface', device).resnet

dataset = datasets.ImageFolder('../../../dat/LFW/test/')
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

# bestimmen des Thresholds ab wann ein Bild als gleich erkannt wird
dist_thresh = 1

# Bestimmen von Accuracy, Precision, Recall und F1 aus der Confusion-Matrix
tp, fp, fn, tn = confusion_matrix(dist_thresh, embeddings, classes, names, dists)
accuracies, precicions, recalls = classification_meassures(tp, fp, fn, tn, classes)

accuracy = np.nanmean(np.array(list(accuracies.values())))
precicion = np.nanmean(np.array(list(precicions.values())))
recall = np.nanmean(np.array(list(recalls.values())))
f1 = 2 * (precicion * recall) / (precicion + recall)

print('Accuracy {:.2f}'.format(accuracy))
print('Precision {:.2f}'.format(precicion))
print('Recall {:.2f}'.format(recall))
print('F1-Score {:.2f}'.format(f1))

# Plotten der ROC-Kurve
# th = np.linspace(0, 1.5, 100)
# tpr, fpr = compute_tpr_fpr(th, classes, embeddings, names, dists)
# plt.plot(fpr, tpr)
# plt.show()

# DBSCAN Test
db = DBSCAN(eps=1, min_samples=1).fit(embeddings)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
labels_true = names

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('DBSCAN Results:')
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))

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

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
