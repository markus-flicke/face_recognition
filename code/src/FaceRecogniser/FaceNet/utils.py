from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from operator import itemgetter
import numpy as np


class Model:
    ''' Für den Parameter model muss entweder der String "vggface2" oder "casia-webface" angegeben werden '''

    def __init__(self, model, device):
        self.resnet = InceptionResnetV1(pretrained=model).eval().to(device)


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


def confusion_matrix(threshold, embeddings, classes, names, dists):
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


def classification_meassures(tp, fp, fn, tn, classes):
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


def compute_tpr_fpr(thresh, classes, embeddings, names, dists):
    tpr = dict(zip(classes, np.zeros(len(classes))))
    tprs = []
    fpr = tpr.copy()
    fprs = []

    for t in thresh:
        tp, fp, fn, tn = confusion_matrix(t, embeddings, classes, names, dists)
        for key in tpr:
            tpr[key] = tp[key] / (tp[key] + fn[key])
            tprs.append(np.nanmean(np.array(list(tpr.values()))))
            fpr[key] = fp[key] / (fp[key] + tn[key])
            fprs.append(np.nanmean(np.array(list(fpr.values()))))

    return tprs, fprs
