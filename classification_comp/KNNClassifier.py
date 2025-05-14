import numpy as np
import pandas as pd

class KNNClassifier:
    def __init__(self, k=3, metric=2):
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []

        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1, ord=self.metric)
            nns = np.argsort(distances)[:self.k]
            nn_labels = self.y_train[nns]

            values, counts = np.unique(nn_labels, return_counts=True)
            majority_label = values[np.argmax(counts)]

            y_pred.append(majority_label)

        return np.array(y_pred)
