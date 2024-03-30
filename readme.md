# Python Image Classification Project

This project is a simple image classification application using K-Nearest Neighbors (KNN) algorithm. The project is implemented in Python and uses the scikit-learn library for the KNN classifier.

## Project Structure

The project consists of the following main components:

- `DataManager.py`: This file contains the `DataManager` class which is responsible for handling the image data. It includes methods for reading images from a directory, converting them to grayscale, and resizing them.

- `knn.py`: This file contains the `KNNClassifier` class which is responsible for training the KNN classifier, making predictions, and saving/loading the trained model.

- `main.py`: This is the main entry point of the application. It checks if a trained model exists, if not, it trains a new model and saves it. Then, it uses the model to make predictions on the test data.

## Dependencies

The project has the following dependencies:

- scikit-learn
- numpy
- PIL

These can be installed using pip:

```bash
pip install -r requirements.txt
```

You can add the information about the new command-line arguments to the `Usage` section of your `README.md` file. Here's how you can update it:

## Usage

### Args

- `--train` : Path to the training dataset
- `--test` : Path to the testing dataset
- `--no-cache` : Ignore any previously saved classifiers and train a new one

### Exemple
```bash
python main.py --train path_to_training_dataset --test path_to_testing_dataset
```

Replace `path_to_training_dataset` and `path_to_testing_dataset` with the actual paths to your datasets.

```bash
python main.py
```