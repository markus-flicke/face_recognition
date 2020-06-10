import unittest

from DataLoader import DataLoader
from main import extrac_all_faces_from_all_albums
from src.FaceRecogniser.VGG2.vgg2 import get_vgg_embeddings
from src.FaceRecogniser.FaceNet.facenet_A2 import get_faceNet_embeddings
from src.metrics import dist_matrix_euclid, threshold_metrics, dist_matrix_cosine


class Metric_Test(unittest.TestCase):
    """
    TODO: Write metric tests for the other networks
    TODO: Implement Clustering Metrics
    """

    def setUp(self):
        # extrac_all_faces_from_all_albums()
        pass

    def test_vgg(self):
        face_paths, labels = DataLoader().load_A2()

        # vgg_embeddings_resnet = get_vgg_embeddings(face_paths,'resnet50')
        #
        # # TODO: same procedure to get embeddings with the other networks
        #
        # dists_res = dist_matrix_euclid(vgg_embeddings_resnet)
        #
        # print('\nScore for VGG Resnet\n')
        # threshold_metrics(900, dists_res, labels, vgg_embeddings_resnet)

        vgg_embeddings_senet = get_vgg_embeddings(face_paths, 'senet50')

        # dists_senet = dist_matrix_euclid(vgg_embeddings_senet)

        # print('\nScore for VGG Senet\n')

        # threshold_metrics(4400, dists_senet, labels, vgg_embeddings_senet)

        print('\nScore for VGG Senet cosine\n')
        dists_senet_cos = dist_matrix_cosine(vgg_embeddings_senet)
        threshold_metrics(0.05, dists_senet_cos, labels, vgg_embeddings_senet)

    def test_faceNet(self):
        face_paths, labels = DataLoader().load_A2()

        faceNet_embeddings = get_faceNet_embeddings(face_paths, 'vggface2')

        dists = dist_matrix_euclid(faceNet_embeddings)

        print('Score for FaceNet')
        threshold_metrics(0.2, dists, labels, faceNet_embeddings)


unittest.main()
