import os
import numpy as np
from DataManager import DataManager
from knn import KNNClassifier
from network import Network
from sklearn.model_selection import train_test_split


def main():
    args = args_parser()
    create_classifier_folders(args)
    if args.knn:
        run_knn(args)
    if args.neuronal:
        run_network(args)


def create_classifier_folders(args):
    if not os.path.exists("classifiers"):
        os.makedirs("classifiers")
    if not os.path.exists("classifiers/network"):
        os.makedirs("classifiers/network")
    if not os.path.exists("classifiers/knn"):
        os.makedirs("classifiers/knn")


def run_knn(args):
    classifier_path = "classifiers/knn/" + "classifier_" + args.train.replace("/", "_") + ".pkl"
    if not check_if_file_exists(classifier_path) or args.no_cache:
        print("Training classifier KNN...")
        train_and_save(args.train, classifier_path)
    else:
        print("Classifier already trained. Using existing classifier.")
    print("Testing KNN classifier...")
    test(args.test, classifier_path, args.confusion_matrix)


def run_network(args):
    classifier_path = "classifiers/network/" + "classifier_" + args.train.replace("/", "_") + ".keras"
    if not check_if_file_exists(classifier_path) or args.no_cache:
        print("Training classifier Neuronal Network...")
        train_and_save_network(args.train, classifier_path, args.neuronal)
    else:
        print("Classifier already trained. Using existing classifier.")
    print("Testing Neuronal classifier ...")
    test_network(args.test, classifier_path)


def test(dataset_path, classifier, display_confusion_matrix):
    classifier = KNNClassifier.load_self(classifier)

    test_data_dict = DataManager.get_data(dataset_path)
    test_data, test_labels = KNNClassifier.prepare_data(test_data_dict)
    classifier.predict(test_data, test_labels, display_confusion_matrix)


def test_network(dataset_path, classifier):
    classifier = Network.load_self(classifier)
    test_data_dict = DataManager.get_data(dataset_path)
    test_data, test_labels = classifier.load_and_preprocess_data(test_data_dict)
    _, _, label_encoder = Network.encode_labels(test_labels)
    Network.evaluate_model(classifier.model, label_encoder, test_labels, test_data)


def train_and_save(datas_path, classifier_path):
    classifier = KNNClassifier()

    train_data_dict = DataManager.get_data(datas_path)
    train_data, train_labels = KNNClassifier.prepare_data(train_data_dict)

    classifier.train(train_data, train_labels)
    classifier.save_self(classifier_path)


def train_and_save_network(dataset_path, classifier_path, epochs):
    classifier = Network()

    train_data_dict = DataManager.get_data(dataset_path)
    train_data, train_labels = classifier.load_and_preprocess_data(train_data_dict)

    # Encode labels
    integer_encoded_labels, onehot_encoded_labels, label_encoder = classifier.encode_labels(train_labels)

    # Split data
    train_images, test_images, train_labels, test_labels = train_test_split(
        train_data, onehot_encoded_labels, test_size=0.2, random_state=42)

    # Build and train model
    num_classes = len(np.unique(integer_encoded_labels))
    model = classifier.build_model(num_classes)
    classifier.train_model(model, train_images, train_labels, test_images, test_labels, epochs)
    classifier.save_self(classifier_path)
    model.save(classifier_path)


def check_if_file_exists(file_path):
    return os.path.isfile(file_path)


def args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="KNN Classifier")
    parser.add_argument("--train", type=str, help="Path to training dataset")
    parser.add_argument("--test", type=str, help="Path to testing dataset")
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore any previously saved classifiers and train a new one')
    parser.add_argument('--confusion-matrix', action='store_true',
                        help='Display confusion matrix after testing the classifier')
    parser.add_argument('--knn', action='store_true',
                        help='Use KNN classifier')
    parser.add_argument('--neuronal', type=int,
                        help='Use Neuronal Network classifier with the specified epochs')
    if not parser.parse_args().train or not parser.parse_args().test:
        parser.error("Please provide both training and testing dataset paths")
        exit(1)
    if not parser.parse_args().knn and not parser.parse_args().neuronal:
        parser.error("Please provide a classifier to use")
        exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
