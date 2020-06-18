import unittest

from src.DataLoader import DataLoader
from src.FaceRecogniser.ResNet34.ResNet34_A2 import get_ResNet_embeddings
from src.FaceRecogniser.VGG2.vgg2 import get_vgg_embeddings
from src.FaceRecogniser.FaceNet.facenet_A2 import get_faceNet_embeddings
from src.Test.metrics import dist_matrix_euclid, threshold_metrics, dist_matrix_cosine, get_best_threshold


class Threshold_Test(unittest.TestCase):

    def setUp(self):
        # extrac_all_faces_from_all_albums()
        pass

    def test_vgg_A2(self):
        face_paths, labels = DataLoader().load_A2()
        vgg_embeddings_senet = get_vgg_embeddings(face_paths, 'senet50')

        dists_senet_cos = dist_matrix_cosine(vgg_embeddings_senet)
        best_threshold = get_best_threshold(dists_senet_cos, labels, vgg_embeddings_senet)
        print('\nScore for VGG Senet cosine\n')
        threshold_metrics(best_threshold, dists_senet_cos, labels,
                          vgg_embeddings_senet, True)
        # classes = np.sort(list(dict.fromkeys(labels)))
        # tpr, fpr = compute_tpr_fpr(np.linspace(0,1,1000),classes, vgg_embeddings_senet, labels, dists_senet_cos)
        # plt.plot(fpr, tpr)
        # plt.show()

    def test_vgg_LFW(self):
        face_paths, labels, d1, d2 = DataLoader().load_lfw()
        vgg_embeddings_senet = get_vgg_embeddings(face_paths, 'senet50')
        dists_senet_cos = dist_matrix_cosine(vgg_embeddings_senet)
        best_threshold = get_best_threshold(dists_senet_cos, labels, vgg_embeddings_senet)
        print('\nScore for VGG Senet cosine\n')
        threshold_metrics(best_threshold, dists_senet_cos, labels,
                          vgg_embeddings_senet, True)

    def test_faceNet(self):
        face_paths, labels = DataLoader().load_A2()

        faceNet_embeddings = get_faceNet_embeddings(face_paths, 'vggface2')

        dists = dist_matrix_euclid(faceNet_embeddings)

        print('Threshold Approach Metrics: FaceNet')
        threshold_metrics(0.2, dists, labels, faceNet_embeddings)

    def test_ResNet34(self):
        face_paths, labels = DataLoader().load_A2()
        resNet_embeddings, labels = get_ResNet_embeddings(face_paths, labels)

        dists = dist_matrix_euclid(resNet_embeddings)

        print('Threshold Approach Metrics: ResNet34 on A2')
        threshold_metrics(0.2, dists, labels, resNet_embeddings)

    def test_ResNet34_LFW(self):
        face_paths, labels, d1, d2 = DataLoader().load_lfw()
        resNet_embeddings, labels = get_ResNet_embeddings(face_paths, labels)

        dists = dist_matrix_euclid(resNet_embeddings)

        print('Threshold Approach Metrics: ResNet34 on LFW')
        threshold_metrics(0.2, dists, labels, resNet_embeddings)
