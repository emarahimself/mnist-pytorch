import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import time
import pca
import initializer
from dataset import *


class HyperParameters:
    epochs = 5000
    lr = 0.001  # learning rate
    lmda = 0.1  # regularization penalty term
    l1 = False
    l2 = False

class Network:
    # Network Architecture
    n1 = 16
    n2 = 12
    n3 = None # To be set explicitly in __init__ depending on number of categories

    def __init__(self, dataset: DatasetBase, hyperparameters: HyperParameters):
        self.dataset = dataset
        self.x_train = dataset.x_train
        self.x_test = dataset.x_test
        self.y_train = dataset.y_train
        self.y_test = dataset.y_test
        self.n3 = dataset.classes

        self.epochs = hyperparameters.epochs
        self.lr = hyperparameters.lr
        self.lmda = hyperparameters.lmda
        self.l1 = hyperparameters.l1
        self.l2 = hyperparameters.l2

        # First Layer
        self.input_dim = dataset.features
        self.w1, self.b1 = initializer.random_init(self.n1, self.input_dim)

        # Second layer
        self.w2, self.b2 = initializer.random_init(self.n2, self.n1)

        # Third layer
        self.w3, self.b3 = initializer.random_init(self.n3, self.n2)

    def forward(self, x):
        z1 = (torch.tensor(x, dtype=torch.float32) @ self.w1.t()) + self.b1.t()
        h1 = torch.relu(z1)
        z2 = (h1 @ self.w2.t()) + self.b2.t()
        h2 = torch.sigmoid(z2)
        z3 = (h2 @ self.w3.t()) + self.b3.t()
        out = torch.softmax(z3, dim=-1)
        return out

    def cost(self, out, y):
        A = torch.tensor(0.0)
        B = torch.tensor(0.0)
        if self.l2:
            A = torch.sum(torch.pow(self.w1, 2)) + torch.sum(torch.pow(self.w2, 2)) + torch.sum(torch.pow(self.w3, 2))
        if self.l1:
            B = torch.sum(torch.abs(self.w1)) + torch.sum(torch.abs(self.w2)) + torch.sum(torch.abs(self.w3))

        y = torch.tensor(y, dtype=torch.float32)
        loss = (y * torch.log(out)).sum(dim=1)
        return -loss.mean() + (self.lmda * A) + (self.lmda * B)

    def train(self, bs=None, verbose=True):
        train_history = {
            'loss': [],
            'train_acc': [],
            'test_acc': []
        }

        for epoch in range(self.epochs):
            # Mini-batch
            if bs is not None:
                idx = random.sample(range(0, self.dataset.train_size), bs)
                x_train_mini = self.x_train[idx]
                y_train_mini = self.y_train[idx]
            else:
                x_train_mini = self.x_train
                y_train_mini = self.y_train

            out = self.forward(x_train_mini)
            cost = self.cost(out, y_train_mini)
            cost.backward()
            params = [self.w1, self.w2, self.w3,
                      self.b1, self.b2, self.b3]

            for param in params:
                param.data -= self.lr * param.grad
                param.grad.data.zero_()

            train_accuracy = self.accuracy(out, y_train_mini)
            test_accuracy = self.accuracy(self.forward(self.x_test), self.y_test)
            train_history['loss'].append(cost.item())
            train_history['train_acc'].append(train_accuracy)
            train_history['test_acc'].append(test_accuracy)

            if verbose:
                print(f'Epoch: {epoch}: train_accuracy= {train_accuracy}, test_accuracy={test_accuracy} '
                      f'and loss= {cost.item()}')
        return train_history

    @staticmethod
    def accuracy(out, y):
        out = (out > 0.5).float()
        out_cats = np.argmax(out, axis=1)
        return accuracy_score(np.argmax(y, axis=1), out_cats)

    def model(self, bs=None, plot=True, verbose=True):
        train_history = self.train(bs, verbose)
        if plot:
            ep = range(self.epochs)
            plt.plot(ep, train_history['loss'], label='Loss')
            plt.plot(ep, train_history['train_acc'], label='Train Accuracy')
            plt.plot(ep, train_history['test_acc'], label='Test Accuracy')
            plt.legend()
            plt.show()
        return train_history

    def predict(self, x):
        y = self.forward(x)
        predictions = torch.argmax(y, dim=1)
        return predictions.detach().numpy()


if __name__ == '__main__':
    hp = HyperParameters()
    hp.epochs = 5000
    hp.lr = 0.1
    hp.lmda = 0.5
    hp.l2 = False
    batch_size = 1000

    nn = Network(pca.PCADataset(MNISTDataset()).dataset(), hp)
    then = time.time()
    history = nn.model(bs=batch_size, plot=True, verbose=True)
    now = time.time()

    print(f'With PCA, train time = {now - then} seconds')
    print('Train Accuracy: {}'.format(history['train_acc'][nn.epochs - 1]))
    print('Test Accuracy: {}'.format(history['test_acc'][nn.epochs - 1]))