import pickle

from sklearn import neighbors, metrics
import numpy as np


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

    def predict(self, test_data, test_labels):
        predicted = self.knn.predict(test_data)
        print("Classification report for classifier %s:\n%s\n"
              % (self.knn, metrics.classification_report(test_labels, predicted)))

    def save_self(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_self(path):
        with open(path, "rb") as f:
            return pickle.load(f)
