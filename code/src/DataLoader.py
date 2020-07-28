import os
import random

import pandas as pd
from src import Config
from sklearn import preprocessing


class DataLoader():

    def load_A2(self):
        df = pd.read_csv(os.path.join(Config.A2_LABELS_PATH))
        return [os.path.join(Config.A2_EXTRACTED_FACES_PATH, filename) for filename in df.values[:, 0]], list(
            df.values[:, 1])

    def load_lfw(self, N_train=100, N_test=10, lfw_type=Config.LFW_PATH):
        """
        Loads the lfw dataset with a specified number of training and test samples.
        The resulting data has only images in the test set whose class also appears at least once during training.
        :param N_train:
        :param N_test:
        :return: training_filepaths, training_labels, test_filepaths, test_labels
        """
        filepaths = []
        labels = []

        for label in os.listdir(lfw_type):
            folder_filenames = os.listdir(os.path.join(lfw_type, label, 'extracted_faces'))
            folder_filepaths = [os.path.join(lfw_type, label, 'extracted_faces', filename) for filename in
                                folder_filenames]
            labels.extend([label] * len(folder_filepaths))
            filepaths.extend(folder_filepaths)

        le = preprocessing.LabelEncoder()
        labels_as_id = le.fit_transform(labels)

        c = list(zip(filepaths, labels_as_id))
        random.shuffle(c)
        filepaths, labels_as_id = zip(*c)
        training_filepaths = filepaths[:N_train]
        training_labels = labels_as_id[:N_train]
        test_filepaths, test_labels = zip(
            *[(filepaths[i], labels_as_id[i]) for i in range(80, len(filepaths)) if labels_as_id[i] in training_labels][
             :N_test])
        return training_filepaths, training_labels, test_filepaths, test_labels

    def load_lfw_complete(self):
        filepaths = []
        labels = []

        for label in os.listdir(Config.LFW_PATH):
            folder_filenames = os.listdir(os.path.join(Config.LFW_PATH, label, 'extracted_faces'))
            folder_filepaths = [os.path.join(Config.LFW_PATH, label, 'extracted_faces', filename) for filename in
                                folder_filenames]
            labels.extend([label] * len(folder_filepaths))
            filepaths.extend(folder_filepaths)

        le = preprocessing.LabelEncoder()
        labels_as_id = le.fit_transform(labels)

        return filepaths, labels_as_id
