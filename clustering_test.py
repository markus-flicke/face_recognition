import unittest

from DataLoader import DataLoader
from clustering import cluster_predictions, evaluate_clustering
from src.FaceRecogniser.FaceNet.facenet_A2 import get_faceNet_embeddings
from src.FaceRecogniser.ResNet34.ResNet34_A2 import get_ResNet_embeddings


class ClusteringTest(unittest.TestCase):
    def test_faceNet(self):
        face_paths, labels = DataLoader().load_A2()
        faceNet_embeddings = get_faceNet_embeddings(face_paths, 'vggface2')
        predictions = cluster_predictions(faceNet_embeddings)
        print('Clustering Metrics: FaceNet')
        evaluate_clustering(labels, predictions)

    def test_resnet_A2(self):
        face_paths, labels = DataLoader().load_A2()
        resNet_embeddings, labels = get_ResNet_embeddings(face_paths, labels)
        predictions = cluster_predictions(resNet_embeddings)
        print('Clustering Metrics: ResNet34 on A2')
        evaluate_clustering(labels, predictions)

    def test_resnet_LFW(self):
        face_paths, labels, d1, d2 = DataLoader().load_lfw()
        resNet_embeddings, labels = get_ResNet_embeddings(face_paths, labels)
        predictions = cluster_predictions(resNet_embeddings)
        print('Clustering Metrics: ResNet34 on LFW')
        evaluate_clustering(labels, predictions)