import unittest

from src.DataLoader import DataLoader
from src.FaceRecogniser.clustering import cluster_predictions, evaluate_clustering, evaluate_best_threshold
from src.FaceRecogniser.FaceNet.facenet_A2 import get_faceNet_embeddings
from src.FaceRecogniser.ResNet34.ResNet34_A2 import get_ResNet_embeddings
from src.FaceRecogniser.VGG2.vgg2 import get_vgg_embeddings
import src.Test.plotter as plotter


class ClusteringTest(unittest.TestCase):

    def test_faceNet_A2(self):
        face_paths, labels = DataLoader().load_A2()
        faceNet_embeddings = get_faceNet_embeddings(face_paths, 'casia-webface')
        best_threshold, best_f1 = evaluate_best_threshold(faceNet_embeddings, labels, 'cosine')
        predictions = cluster_predictions(faceNet_embeddings, best_threshold, 1, 'cosine')
        plotter.plotPredictions(predictions, face_paths, max_images_per_plot=20)
        print('Clustering Metrics: FaceNet on A2')
        evaluate_clustering(labels, predictions)

    def test_faceNet_LFW(self):
        face_paths, labels, d1, d2 = DataLoader().load_lfw()
        faceNet_embeddings = get_faceNet_embeddings(face_paths, 'vggface2')
        best_threshold, best_f1 = evaluate_best_threshold(faceNet_embeddings, labels, 'cosine')
        predictions = cluster_predictions(faceNet_embeddings, best_threshold, 1, 'cosine')
        # plotter.plotPredictions(predictions, face_paths, max_images_per_plot=10)
        print('Clustering Metrics: FaceNet on LFW')
        evaluate_clustering(labels, predictions)

    def test_resnet_A2(self):
        face_paths, labels = DataLoader().load_A2()
        resNet_embeddings, labels, face_paths = get_ResNet_embeddings(face_paths, labels)
        best_threshold, best_f1 = evaluate_best_threshold(resNet_embeddings, labels, 'cosine')
        predictions = cluster_predictions(resNet_embeddings,best_threshold,1,'cosine')
        plotter.plotPredictions(predictions, face_paths)
        print('Clustering Metrics: ResNet34 on A2')
        evaluate_clustering(labels, predictions)

    def test_resnet_LFW(self):
        face_paths, labels, d1, d2 = DataLoader().load_lfw()
        resNet_embeddings, labels, face_paths = get_ResNet_embeddings(face_paths, labels)
        best_threshold, best_f1 = evaluate_best_threshold(resNet_embeddings, labels, 'cosine')
        predictions = cluster_predictions(resNet_embeddings,best_threshold,1,'cosine')
        print('Clustering Metrics: ResNet34 on LFW')
        evaluate_clustering(labels, predictions)

    def test_vgg_A2(self):
        face_paths, labels = DataLoader().load_A2()
        vgg_embeddings = get_vgg_embeddings(face_paths, 'resnet50')
        best_threshold, best_f1 = evaluate_best_threshold(vgg_embeddings, labels, 'cosine')
        predictions = cluster_predictions(vgg_embeddings,best_threshold,1,'cosine')
        plotter.plotPredictions(predictions, face_paths)
        print('Clustering Metrics: VGG2 on A2')
        evaluate_clustering(labels, predictions)

    def test_vgg_lfw(self):
        face_paths, labels, d1, d2 = DataLoader().load_lfw()
        vgg_embeddings = get_vgg_embeddings(face_paths, 'resnet50')
        best_threshold, best_f1 = evaluate_best_threshold(vgg_embeddings, labels, 'cosine')
        print(best_threshold)
        predictions = cluster_predictions(vgg_embeddings, best_threshold, 1, 'cosine')
        print('Clustering Metrics: VGG2 on LFW')
        evaluate_clustering(labels, predictions)
