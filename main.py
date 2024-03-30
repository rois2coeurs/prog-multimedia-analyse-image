import os

from DataManager import DataManager
from knn import KNNClassifier


def main():
    args = args_parser()
    classifier_path = "classifiers/" + "classifier_" + args.train.replace("/", "_") + ".pkl"

    if not check_if_file_exists(classifier_path) or args.no_cache:
        print("Training classifier...")
        train_and_save(args.train, classifier_path)
    else:
        print("Classifier already trained. Using existing classifier.")

    print("Testing classifier...")
    test(args.test, classifier_path)


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


def args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="KNN Classifier")
    parser.add_argument("--train", type=str, help="Path to training dataset")
    parser.add_argument("--test", type=str, help="Path to testing dataset")
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore any previously saved classifiers and train a new one')
    if not parser.parse_args().train or not parser.parse_args().test:
        parser.error("Please provide both training and testing dataset paths")
        exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
