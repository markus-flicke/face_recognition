from sklearn.cluster import DBSCAN


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




def cluster_predictions(feature_vectors, max_dist=0.3, min_samples=1, metric='euclidean'):
    clt = DBSCAN(eps=max_dist, min_samples=min_samples, metric=metric)
    return clt.fit_predict(feature_vectors)


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
