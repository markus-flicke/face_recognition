from operator import itemgetter
import numpy as np
from scipy.spatial.distance import cosine

def threshold_metrics(dist_thresh, dist_matrix, labels, embeddings, verbose = True):
    classes = np.sort(list(dict.fromkeys(labels)))

    # Bestimmen von Accuracy, Precision, Recall und F1 aus der Confusion-Matrix
    tp, fp, fn, tn = confusion_matrix(dist_thresh, embeddings, classes, labels, dist_matrix)
    accuracies, precicions, recalls = classification_metrics(tp, fp, fn, tn, classes)

    accuracy = np.nanmean(np.array(list(accuracies.values())))
    precision = np.nanmean(np.array(list(precicions.values())))
    recall = np.nanmean(np.array(list(recalls.values())))
    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

    if verbose:
        print('Accuracy {:.2f}'.format(accuracy))
        print('Precision {:.2f}'.format(precision))
        print('Recall {:.2f}'.format(recall))
        print('F1-Score {:.2f}'.format(f1))
    return accuracy,precision, recall, f1


def dist_matrix_euclid(embeddings):
    dists = [[(np.linalg.norm(e1 - e2) ** 2) for e2 in embeddings] for e1 in embeddings]
    return dists


def get_best_threshold(dist_mat, labels, embeddings):
    best_f1 = 0
    best_threshold = 0
    for i in range(1, 1000):
        threshold = i * 0.001
        accuracy, precision, recall, f1 = threshold_metrics(threshold, dist_mat, labels,
                                                            embeddings, False)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def dist_matrix_cosine(embeddings):
    dists = [[cosine(e2,e1) for e2 in embeddings] for e1 in embeddings]
    return dists

def get_top3_predictions(dist, labels):
    """
    Berechnet die Top 3 predictions
    :param dist: List Distanzmatrix NxN
    :param labels: List Label der Testdatensätze 1xN
    :return: List of tupel (ground_truth_label, [top_3_label]) 1xN
    """
    top = []
    for row in dist:
        dist_class = zip(row, labels)
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


def confusion_matrix(threshold, embeddings, classes, labels, dists):
    tp = dict(zip(classes, np.zeros(len(classes))))
    tn = tp.copy()
    fp = tp.copy()
    fn = tp.copy()
    for row in range(len(embeddings)):
        for col in range(len(embeddings)):
            current_class = labels[row]
            if row != col:
                if dists[row][col] <= threshold:
                    if current_class == labels[col]:
                        tp[current_class] += 1
                    else:
                        fp[current_class] += 1
                else:
                    if current_class == labels[col]:
                        fn[current_class] += 1
                    else:
                        tn[current_class] += 1
    return tp, fp, fn, tn


'''
Berechnet Accuracy, Precision und Recall für jede Klasse

:tp, fp, fn, tn: Dict {class: value} für tp, fp, tn, fn

:return: Dict {class: value} für Accuracy, Precicion und Recall 
'''


def classification_metrics(tp, fp, fn, tn, classes):
    accuracies = dict(zip(classes, np.zeros(len(classes))))
    precicions = accuracies.copy()
    recalls = accuracies.copy()
    for key in accuracies:
        accuracies[key] = (tp[key] + tn[key]) / (fp[key] + fn[key] + tp[key] + tn[key]) if (fp[key] + fn[key] + tp[key] + tn[key]) != 0 else 0
        precicions[key] = tp[key] / (tp[key] + fp[key]) if (tp[key] + fp[key]) != 0 else 0
        recalls[key] = tp[key] / (tp[key] + fn[key]) if (tp[key] + fn[key]) != 0 else 0
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
            tpr[key] = tp[key] / (tp[key] + fn[key]) if (tp[key] + fn[key])!=0 else 0
            tprs.append(np.nanmean(np.array(list(tpr.values()))))
            fpr[key] = fp[key] / (fp[key] + tn[key]) if (fp[key] + tn[key]) != 0 else 0
            fprs.append(np.nanmean(np.array(list(fpr.values()))))

    return tprs, fprs
