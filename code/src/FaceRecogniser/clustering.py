from sklearn.cluster import DBSCAN
import numpy as np


def evaluate_best_threshold(embeddings, labels, metric):
    best_threshold = 0
    best_f1 = 0
    for i in range(1, 1000):
        threshold = 0.001 * i
        predictions = cluster_predictions(embeddings, threshold, 1, metric)
        tp,fp,fn,tn = evaluate_clustering(labels, predictions, False)
        f1_val = f1(tp,fp,fn)
        if f1_val > best_f1:
            best_threshold = threshold
            best_f1 = f1_val
    return best_threshold, best_f1


def get_closest_clusters(embeddings, clusters):
    cluster_means = get_cluster_means(embeddings, clusters)
    ## computes list of closest clusters based on cluster indices
    result = []
    for idx1, cluster_mean1 in enumerate(cluster_means):
        closest_dist = None
        closest_indice = None
        for idx2, cluster_mean2 in enumerate(cluster_means):
            dist = np.linalg.norm(cluster_mean2 - cluster_mean1)
            if idx1 != idx2 and (closest_dist == None or closest_dist>dist):
                closest_indice = idx2
                closest_dist = dist
        result.append(closest_indice)
    return result


def cluster_predictions(feature_vectors, max_dist=0.3, min_samples=1, metric='euclidean'):
    clt = DBSCAN(eps=max_dist, min_samples=min_samples, metric=metric)
    return clt.fit_predict(feature_vectors)

def get_cluster_means(embeddings, clusters):
    result = []
    for i in range(0, np.amax(clusters)+1):
        cluster_embedding_indices = np.where(clusters == i)
        cluster_embeddings = embeddings[cluster_embedding_indices]
        mean = np.mean(cluster_embeddings, axis=0)
        result.append(mean)
    return result

def jaccard(tp, fp, fn):
    return tp / (tp + fp + fn)


def accuracy(tp, fp, fn, tn):
    return 0 if (tp + fp + fn + tn) == 0 else  (tp + tn) / (tp + fp + fn + tn)


def precision(tp, fp):
    return 0 if tp + fp == 0 else tp / (tp + fp)


def recall(tp, fn):
    return 0 if (tp + fn) == 0 else tp / (tp + fn)


def f1(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 0 if (p + r) == 0 else 2 * p * r / (p + r)

def evaluate_clustering(y, pred, verbose=True):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for true_label, pred_label in zip(y, pred):
        true_idx = {i for i, t in enumerate(y) if t == true_label}
        pred_idx = {i for i, p in enumerate(pred) if p == pred_label}
        tp += len(pred_idx.intersection(true_idx)) - 1
        fp += len(pred_idx.difference(true_idx))
        fn += len(true_idx.difference(pred_idx))
        tn += len(y) - len(pred_idx.intersection(true_idx)) - len(pred_idx.difference(true_idx)) - len(
            true_idx.difference(pred_idx))
    if verbose:
        print(f'Accuracy:  {accuracy(tp, fp, fn, tn):>8.2%}')
        print(f'Jaccard:   {jaccard(tp, fp, fn):>8.2%}')
        print(f'Precision: {precision(tp, fp):>8.2%}')
        print(f'Recall:    {recall(tp, fn):>8.2%}')
        print(f'F1:        {f1(tp, fp, fn):>8.2%}')
        print()
        print(f'{"":<12}{"Positive":<12}{"Negative":<12}')
        print(f'{"True":<12}{tp:<12}{tn:<12}')
        print(f'{"False":<12}{fp:<12}{fn:<12}')
    return tp,fp,fn,tn

def get_clusters(embeddings, threshold, min_cluster_size):
    threshold = threshold / 100
    predictions = cluster_predictions(embeddings, threshold, min_cluster_size, 'cosine')
    predictions_clean = predictions[predictions != -1]
    cluster_count = np.unique(predictions_clean).shape[0]
    avg_imgs_per_cluster = 0
    if (cluster_count!=0):
        avg_imgs_per_cluster = predictions_clean.shape[0] / cluster_count
    closest_clusters = get_closest_clusters(embeddings, predictions_clean)
    return predictions, cluster_count, avg_imgs_per_cluster, closest_clusters