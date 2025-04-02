import numpy as np

class Perceptron():
    def __init__(self, learning_rate: float, n_iter: int):
        self.learning_rate = learning_rate 
        self.n_iter = n_iter

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features) * 0.01
        #self.weights = np.zeros(1 + n_features)

        self.bias = 0

        for _ in range(self.n_iter):
            for idx, x_j in enumerate(X):
                linear_output = np.dot(x_j, self.weights) + self.bias
                y_predicted = 1 if linear_output > 0 else 0 #Heaviside step func

                if y_predicted == y[idx]:
                    continue

                update = self.learning_rate * (y[idx] - y_predicted)

                self.weights += update * x_j
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)