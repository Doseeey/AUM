import numpy as np

class Perceptron():
    def __init__(self, learning_rate: float, n_iter: int):
        self.learning_rate = learning_rate 
        self.n_iter = n_iter

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        _, n_features = X.shape
        self.weights = np.zeros(n_features)

        self.bias = 0

        for _ in range(self.n_iter):
            for idx, x_j in enumerate(X):
                linear_output = np.dot(x_j, self.weights) + self.bias
                #y_predicted = 1 if linear_output >= 0 else 0 #Heaviside step func
                y_predicted = 1 / (1 + np.exp(-linear_output))

                # if y_predicted == y[idx]:
                #     continue

                update = self.learning_rate * (y[idx] - y_predicted) * y_predicted * (1 - y_predicted)
                self.weights += update * x_j
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-linear_output))
        return np.where(y_pred > 0.5, 1, 0)
        #return np.where(linear_output >= 0, 1, 0)