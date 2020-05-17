import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from src import Config

def classification_error_view(predictions,
                              test_filepaths,
                              test_labels,
                              training_filepaths,
                              training_labels,
                              fig_name = 'unnamed.png'):
    prediction_mistakes = [(predictions[i], test_filepaths[i], test_labels[i]) for i in range(len(predictions)) if predictions[i] != test_labels[i]]

    n_cols = 5
    # Never let rows < 2, as then pyplot decides to have a 1D list of axes, rather than 2D. (>.<)
    n_rows = min(10, len(prediction_mistakes))

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 10))
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=-0.3)
    fig.suptitle(
        '        Test img          |                            Prediction class                    |                           Target class               ',
        fontsize=20, y=0.6)

    for i, row in enumerate(prediction_mistakes):
        if i == n_rows:
            break
        pred, t_fp, t_l = row
        __row_in_plot(ax[i],
                          pred,
                          t_fp,
                          t_l,
                          training_filepaths,
                          training_labels)
    plt.savefig(fig_name)

def __row_in_plot(ax, prediction_label, test_filepath, test_label, training_filepaths, training_labels):
    for i in range(5):
        ax[i].axis('off')

    img = cv2.imread(test_filepath)
    ax[0].title.set_text(os.path.split(test_filepath)[-1])
    ax[0].imshow(img)

    # Prediction class representation
    training_labels= np.array(training_labels)
    training_filepaths = np.array(training_filepaths)
    prediction_class_idx = np.where(training_labels == prediction_label)
    prediction_class_filepaths = training_filepaths[prediction_class_idx]
    img = cv2.imread(prediction_class_filepaths[0])
    ax[1].title.set_text(os.path.split(prediction_class_filepaths[0])[-1])
    ax[1].imshow(img)

    img = cv2.imread(prediction_class_filepaths[-1])
    ax[2].title.set_text(os.path.split(prediction_class_filepaths[-1])[-1])
    ax[2].imshow(img)

    # Target class representation
    training_labels = np.array(training_labels)
    training_filepaths = np.array(training_filepaths)
    target_class_idx = np.where(training_labels == test_label)
    target_class_filepaths = training_filepaths[target_class_idx]
    img = cv2.imread(target_class_filepaths[0])
    ax[3].title.set_text(os.path.split(target_class_filepaths[0])[-1])
    ax[3].imshow(img)

    img = cv2.imread(target_class_filepaths[-1])
    ax[4].title.set_text(os.path.split(target_class_filepaths[-1])[-1])
    ax[4].imshow(img)