import os
import numpy as np
from PIL import Image


class DataManager:
    @staticmethod
    def get_data(folder):
        dataset = {}
        for root, dirs, files in os.walk(folder):
            for dir in dirs:
                dataset[dir] = DataManager.get_all_data_in_folder(os.path.join(root, dir))
        return dataset

    @staticmethod
    def get_all_data_in_folder(folder):
        result = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                result.append(DataManager.read_image(os.path.join(root, file)))
        return result

    @staticmethod
    def read_image(path, size=(64, 64)):
        img = Image.open(path).convert('L')  # convert image to grayscale
        img = img.resize(size)
        return np.array(img)
