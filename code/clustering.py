from sklearn.cluster import DBSCAN


def cluster(feature_vectors, max_dist, min_samples, metric):
    clt = DBSCAN(metric=metric, eps=max_dist, min_samples=min_samples)
    return clt.fit(feature_vectors)


def calculate_similarity_matrix():
    return 0
