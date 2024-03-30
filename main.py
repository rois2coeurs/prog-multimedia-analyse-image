import os

from DataManager import DataManager
from knn import KNNClassifier


def main():
    training_dataset_path = "datas/trainingSet"
    testing_dataset_path = "datas/testLabeled"
    classifier_path = "classifiers/" + "classifier_" + training_dataset_path.replace("/", "_") + ".pkl"

    if not check_if_file_exists(classifier_path):
        print("Training classifier...")
        train_and_save(training_dataset_path, classifier_path)
    else:
        print("Classifier already trained. Using existing classifier.")

    test(testing_dataset_path, classifier_path)


def test(dataset_path, classifier):
    classifier = KNNClassifier.load_self(classifier)

    test_data_dict = DataManager.get_data(dataset_path)
    test_data, test_labels = KNNClassifier.prepare_data(test_data_dict)
    classifier.predict(test_data, test_labels)


def train_and_save(datas_path, classifier_path):
    classifier = KNNClassifier()

    train_data_dict = DataManager.get_data(datas_path)
    train_data, train_labels = KNNClassifier.prepare_data(train_data_dict)

    classifier.train(train_data, train_labels)
    classifier.save_self(classifier_path)


def check_if_file_exists(file_path):
    return os.path.isfile(file_path)


if __name__ == "__main__":
    main()
