import pickle

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import neighbors, metrics
import numpy as np
from sklearn.metrics import confusion_matrix


class KNNClassifier:
    def __init__(self):
        self.knn = neighbors.KNeighborsClassifier()

    @staticmethod
    def prepare_data(data_dict):
        images = []
        labels = []
        for label, imgs in data_dict.items():
            for img in imgs:
                images.append(img)
                labels.append(label)
        images = np.array(images)
        labels = np.array(labels)

        n_samples = len(images)
        n_features = images[0].shape[0] * images[0].shape[1]  # assuming images are grayscale
        data = images.reshape((n_samples, n_features))

        return data, labels

    def train(self, train_data, train_labels):
        self.knn.fit(train_data, train_labels)

    def predict(self, test_data, test_labels, display_confusion_matrix):
        predicted = self.knn.predict(test_data)
        self.print_classification_report(predicted, test_labels)
        if display_confusion_matrix:
            self.display_confusion_matrix(test_labels, predicted)

    def print_classification_report(self, predicted, test_labels):
        print("Classification report for classifier %s:\n%s\n"
              % (self.knn, metrics.classification_report(test_labels, predicted)))

    @staticmethod
    def display_confusion_matrix(test_labels, predicted):
        cm = confusion_matrix(test_labels, predicted)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

    def save_self(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_self(path):
        with open(path, "rb") as f:
            return pickle.load(f)
