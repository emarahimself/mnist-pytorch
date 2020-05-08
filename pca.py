import dataset
from sklearn.decomposition import PCA
import numpy as np


class PCADataset:
    def __init__(self, ds:dataset.DatasetBase, explained_variance_ratio=0.95):
        self.ds = ds
        m = self.ds.train_size
        m_test = self.ds.test_size
        n = self.ds.features
        self.pca = PCA(explained_variance_ratio)

        X = self.pca.fit_transform(
            np.vstack([ds.x_train, ds.x_test])
        )
        ds.x_train = X[0:m,:]
        ds.x_test = X[m:,:]

    def dataset(self):
        return self.ds
