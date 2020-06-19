import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import src.Config as conf


def plotPredictions(predictions, face_paths, min_cluster_size = 2, max_images_per_plot = 5):
    # source https:\\stackoverflow.com\questions\46615554\how-to-display-multiple-images-in-one-figure-correctly\46616645
    labels = np.unique(predictions)
    labels = labels[labels > -1]
    for label in labels:
        image_indices = np.where(predictions == label)[0]
        if image_indices.shape[0] >= min_cluster_size:
            fig = plt.figure()
            rows = 1
            columns = min(image_indices.shape[0], max_images_per_plot)
            ax = []
            for i in range(columns * rows):
                image_indice = image_indices[i]
                img = cv2.imread(face_paths[image_indice])
                ax.append(fig.add_subplot(rows, columns, i + 1))
                curr_ax = ax[-1]
                curr_ax.set_xticks([])
                curr_ax.set_yticks([])
                plt.imshow(img)
        # use ax for further manipulations if necessary
        plt.show()