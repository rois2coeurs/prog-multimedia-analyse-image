import numpy as np
from PIL import Image
from keras.src.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Network(object):
    def __init__(self, model=Sequential()):
        self.model = model

    @staticmethod
    def load_and_preprocess_data(data_dict):
        """
        Load dataset and preprocess images and labels.
        """
        samples, labels = [], []

        for label, images in data_dict.items():
            for image in images:
                samples.append(image)
                labels.append(label)

        samples = np.array(samples).astype('float32') / 255.0
        samples = samples.reshape((-1, 64, 64, 1))  # Reshape for CNN input
        return samples, labels

    @staticmethod
    def encode_labels(labels):
        """
        Encode labels as one-hot vectors.
        """
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        onehot_encoded = to_categorical(integer_encoded)
        return integer_encoded, onehot_encoded, label_encoder

    def build_model(self, num_classes):
        """
        Build and compile the CNN model.
        """
        self.model = Sequential([
            Input(shape=(64, 64, 1)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model

    @staticmethod
    def train_model(model, train_images, train_labels, test_images, test_labels, epochs):
        """
        Train the model with early stopping.
        """
        model.fit(train_images, train_labels, batch_size=32, epochs=epochs,
                  validation_data=(test_images, test_labels))

    @staticmethod
    def preprocess_image(image_path, img_height, img_width):
        """
        Preprocess the image and return it as a numpy array
        :param image_path: path to the image
        :param img_height: height of the image
        :param img_width:  width of the image
        :return: image as a numpy array
        """
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((img_height, img_width))  # Resize the image
        img_array = np.array(img)  # Convert image to array
        img_array = img_array / 255.0  # Normalize the image
        img_array = img_array.reshape((1, img_height, img_width, 1))  # Add the batch dimension
        return img_array

    @staticmethod
    def evaluate_model(model, label_encoder, test_labels, test_images):
        """
        Evaluate the model on individual files in a specified directory,
        calculate the error rate, and also plot the confusion matrix based on the full test set.
        """
        # Make predictions on the test set
        predictions = model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = label_encoder.transform(test_labels)

        # Generate the confusion matrix
        conf_matrix = confusion_matrix(true_classes, predicted_classes)

        # Calculate the error rate
        error_rate = np.mean(predicted_classes != true_classes)
        print("Error Rate on Test Set:", error_rate)

        # Visualize the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def save_self(self, path):
        self.model.save(path)

    @staticmethod
    def load_self(path):
        model = load_model(path)
        network = Network(model=model)
        return network
