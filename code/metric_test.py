import unittest

from DataLoader import DataLoader
from main import extrac_all_faces_from_all_albums
from src.FaceRecogniser.VGG2.vgg2 import get_vgg_embeddings
from src.metrics import dist_matrix, threshold_metrics


class Metric_Test(unittest.TestCase):
    """
    TODO: Write metric tests for the other networks
    TODO: Implement Clustering Metrics
    """
    def setUp(self):
        extrac_all_faces_from_all_albums()

    def test_vgg(self):
        face_paths, labels = DataLoader().load_A2()

        vgg_embeddings = get_vgg_embeddings(face_paths)

        # TODO: same procedure to get embeddings with the other networks

        dists = dist_matrix(vgg_embeddings)

        threshold_metrics(4000, dists, labels, vgg_embeddings)


