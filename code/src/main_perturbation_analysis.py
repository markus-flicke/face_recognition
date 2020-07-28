import matplotlib.pyplot as plt
import src.Config as Config
import numpy as np

from src.DataLoader import DataLoader
from src.image_preprocessing import perturb_images, load_images, save_images
from src.FaceRecogniser.clustering import cluster_predictions, evaluate_clustering, evaluate_best_threshold, accuracy, \
    f1, precision, recall
from src.FaceRecogniser.ResNet34.ResNet34_A2 import get_ResNet_embeddings
from src.FaceRecogniser.VGG2.vgg2 import get_vgg_embeddings
import src.Test.plotter as plotter


def result_resnet_clustering(dop):
    if dop == 0:
        path = Config.LFW_PATH
    elif dop == 1:
        path = Config.LFW_LOW_RES_PATH_step_1
    elif dop == 2:
        path = Config.LFW_LOW_RES_PATH_step_2
    elif dop == 3:
        path = Config.LFW_LOW_RES_PATH_step_3
    elif dop == 4:
        path = Config.LFW_LOW_RES_PATH_step_4
    else:
        print('Degree of perturbation only between 0 and 4')
    face_paths, labels, d1, d2 = DataLoader().load_lfw(lfw_type=path)
    resNet_embeddings, labels, face_paths = get_ResNet_embeddings(face_paths, labels)
    best_threshold, best_f1 = evaluate_best_threshold(resNet_embeddings, labels, 'cosine')
    predictions = cluster_predictions(resNet_embeddings, best_threshold, 1, 'cosine')
    print('Clustering Metrics: ResNet34 on LFW')
    tp, fp, fn, tn = evaluate_clustering(labels, predictions)

    return accuracy(tp, fp, fn, tn), f1(tp, fp, fn), precision(tp, fp), recall(tp, fn)


def result_vgg_clustering(dop):
    if dop == 0:
        path = Config.LFW_PATH
    elif dop == 1:
        path = Config.LFW_LOW_RES_PATH_step_1
    elif dop == 2:
        path = Config.LFW_LOW_RES_PATH_step_2
    elif dop == 3:
        path = Config.LFW_LOW_RES_PATH_step_3
    elif dop == 4:
        path = Config.LFW_LOW_RES_PATH_step_4
    else:
        print('Degree of perturbation only between 0 and 4')
    face_paths, labels, d1, d2 = DataLoader().load_lfw(lfw_type=path)
    vgg_embeddings = get_vgg_embeddings(face_paths, 'resnet50')
    best_threshold, best_f1 = evaluate_best_threshold(vgg_embeddings, labels, 'cosine')
    print(best_threshold)
    predictions = cluster_predictions(vgg_embeddings, best_threshold, 1, 'cosine')
    print('Clustering Metrics: VGG2 on LFW')
    tp, fp, fn, tn = evaluate_clustering(labels, predictions)

    return accuracy(tp, fp, fn, tn), f1(tp, fp, fn), precision(tp, fp), recall(tp, fn)


def plotter(title, acc, f1):
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 4, 5)
    plt.plot(x, acc, 'b--', label='Accuracy')
    plt.plot(x, f1, 'g--', label='F1')
    plt.legend(loc='lower left')
    plt.ylabel('Percentage')
    plt.xlabel('Degree of perturbation')
    plt.xticks(x, ['0', '1', '2', '3', '4'])
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    '''Uncomment to create 4 different low resolution lfw datasets. Takes some time!'''
    # face_images, orig_paths = load_images()
    # perturbed_images_1 = perturb_images(face_images, 1)
    # save_images(perturbed_images_1, orig_paths, Config.LFW_LOW_RES_PATH_step_1)
    # perturbed_images_2 = perturb_images(face_images, 2)
    # save_images(perturbed_images_2, orig_paths, Config.LFW_LOW_RES_PATH_step_2)
    # perturbed_images_3 = perturb_images(face_images, 3)
    # save_images(perturbed_images_3, orig_paths, Config.LFW_LOW_RES_PATH_step_3)
    # perturbed_images_4 = perturb_images(face_images, 4)
    # save_images(perturbed_images_4, orig_paths, Config.LFW_LOW_RES_PATH_step_4)

    acc_resnet, f1_resnet, precision_resnet, recall_resnet = [], [], [], []
    acc_vgg, f1_vgg, precision_vgg, recall_vgg = [], [], [], []
    for i in range(5):
        ar, fr, pr, rr = result_resnet_clustering(i)
        # av, fv, pv, rv = result_vgg_clustering(i)

        acc_resnet.append(ar)
        # acc_vgg.append(av)
        f1_resnet.append(fr)
        # f1_vgg.append(fv)
        precision_resnet.append(pr)
        # precision_vgg.append(pv)
        recall_resnet.append(rr)
        # recall_vgg.append(rv)

    plotter('Decoy of Resnet', acc_resnet, f1_resnet)
    # plotter('Decoy of VGG', acc_vgg, f1_vgg)


