import idx2numpy
from sklearn import preprocessing
from sklearn.datasets import load_iris, load_digits, make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class DatasetBase:
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    def normalize(self):
        raise NotImplementedError

    def binarize_labels(self):
        self.binarizer = preprocessing.LabelBinarizer()
        self.y_train = self.binarizer.fit_transform(self.y_train)
        self.y_test = self.binarizer.fit_transform(self.y_test)

    @property
    def features(self):
        return self.x_train.shape[1]

    @property
    def train_size(self):
        return self.x_train.shape[0]

    @property
    def test_size(self):
        return self.x_test.shape[0]

    @property
    def classes(self):
        return self.y_test.shape[1]


class BlobsDataset(DatasetBase):
    def __init__(self, samples=1000, centers=3, features=20):
        x, y = make_blobs(n_samples=samples, centers=centers, n_features=features)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.25, random_state=1
        )
        self.normalize()
        self.binarize_labels()

    def normalize(self):
        scaler = preprocessing.MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)


class IrisDataset(DatasetBase):
    def __init__(self):
        x, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=1
        )
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.normalize()
        self.binarize_labels()

    def normalize(self):
        scaler = preprocessing.MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)


class MNISTDataset(DatasetBase):
    def __init__(self):
        train_images_path = 'dataset/train-images-idx3-ubyte'
        test_images_path = 'dataset/t10k-images-idx3-ubyte'
        train_labels_path = 'dataset/train-labels-idx1-ubyte'
        test_labels_path = 'dataset/t10k-labels-idx1-ubyte'
        self.x_train = idx2numpy.convert_from_file(train_images_path)
        self.y_train = idx2numpy.convert_from_file(train_labels_path)
        self.x_test = idx2numpy.convert_from_file(test_images_path)
        self.y_test = idx2numpy.convert_from_file(test_labels_path)

        features = 28 * 28
        train_size = self.x_train.shape[0]
        test_size = self.x_test.shape[0]

        self.x_train = self.x_train.reshape(train_size, features)
        self.x_test = self.x_test.reshape(test_size, features)
        self.normalize()
        self.binarize_labels()

    def normalize(self):
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0


class DigitsDataset(DatasetBase):
    def __init__(self):
        x, y = load_digits(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=1
        )
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.normalize()
        self.binarize_labels()

    def normalize(self):
        scaler = preprocessing.MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)
