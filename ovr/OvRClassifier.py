import numpy as np

from Perceptron import Perceptron

class OvRClassifier:
    def __init__(self, learning_rate, n_iter):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.estimators = []

        for cls in self.classes:
            y_binary = (y == cls).astype(int) #Change wine labels to 1 if match current label or 0 otherwise
            estimator = Perceptron(learning_rate=self.learning_rate, n_iter=self.n_iter)
            estimator.fit(X, y_binary)
            self.estimators.append(estimator)

    def predict(self, X):
        scores = np.array([estimator.predict_proba(X) for estimator in self.estimators])
        # self.classes[np.argmax(scores, axis=0)] == np.argmax(scores, axis=0)
        return self.classes[np.argmax(scores, axis=0)]
