import unittest

from DataLoader import DataLoader
from main import extrac_all_faces_from_all_albums
from src.FaceRecogniser.VGG2.vgg2 import get_vgg_embeddings
from src.FaceRecogniser.FaceNet.facenet_A2 import get_faceNet_embeddings
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

        print('Score for VGG')
        threshold_metrics(4000, dists, labels, vgg_embeddings)

    def test_faceNet(self):
        face_paths, labels = DataLoader().load_A2()

        faceNet_embeddings = get_faceNet_embeddings(face_paths, 'vggface2')

        dists = dist_matrix(faceNet_embeddings)

        print('Score for FaceNet')
        threshold_metrics(0.2, dists, labels, faceNet_embeddings)
