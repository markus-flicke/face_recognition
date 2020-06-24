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
        print('\nFace Verification Metrics: VGG2 on A2\n')
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
        print('\nFace Verification Metrics: VGG2 on LFW\n')
        threshold_metrics(best_threshold, dists_senet_cos, labels,
                          vgg_embeddings_senet, True)

    def test_faceNet_A2(self):
        face_paths, labels = DataLoader().load_A2()
        faceNet_embeddings = get_faceNet_embeddings(face_paths, 'casia-webface')
        dists = dist_matrix_cosine(faceNet_embeddings)
        best_threshold = get_best_threshold(dists, labels, faceNet_embeddings)
        print('Best threshold is {}'.format(str(best_threshold)))
        print('\nFace Verification Metrics: FaceNet on A2\n')
        threshold_metrics(best_threshold, dists, labels, faceNet_embeddings)

    def test_faceNet_LFW(self):
        face_paths, labels, d1, d2 = DataLoader().load_lfw()
        faceNet_embeddings = get_faceNet_embeddings(face_paths, 'casia-webface')
        dists = dist_matrix_cosine(faceNet_embeddings)
        best_threshold = get_best_threshold(dists, labels, faceNet_embeddings)
        print('Best threshold is {}'.format(str(best_threshold)))
        print('\nFace Verification Metrics: FaceNet on LFW\n')
        threshold_metrics(best_threshold, dists, labels, faceNet_embeddings)

    def test_ResNet34_A2(self):
        face_paths, labels = DataLoader().load_A2()
        resNet_embeddings, labels, face_paths = get_ResNet_embeddings(face_paths, labels)

        dists = dist_matrix_euclid(resNet_embeddings)
        best_threshold = get_best_threshold(dists, labels, resNet_embeddings)
        print('\nFace Verification Metrics: ResNet34 on A2\n')
        threshold_metrics(best_threshold, dists, labels, resNet_embeddings)

    def test_ResNet34_LFW(self):
        face_paths, labels, d1, d2 = DataLoader().load_lfw()
        resNet_embeddings, labels, face_paths = get_ResNet_embeddings(face_paths, labels)

        dists = dist_matrix_euclid(resNet_embeddings)
        best_threshold = get_best_threshold(dists, labels, resNet_embeddings)
        print('\nFace Verification Metrics: ResNet34 on LFW\n')
        threshold_metrics(best_threshold, dists, labels, resNet_embeddings)
