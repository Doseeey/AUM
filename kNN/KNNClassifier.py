import numpy as np

class KNNClassifier:
    def __init__(self, k=3, metric=2):
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        #Obliczenie dystansu
        #Odnalezienie najblizszych k - argsort
        #Przypisanie etykiet
        
        y_pred = []

        for x in X_test.values:
            distances = np.linalg.norm(self.X_train.values - x, axis=1, ord=self.metric)

            nns = np.argsort(distances)[:self.k] #k najblizszych sasiadow
            #print(nns)

            nn_labels = self.y_train.iloc[nns] #etykiety najblizszych sasiadow

            y_pred.append(nn_labels.mode()[0]) #ustawienie etykiety pojawiajacej sie najczesciej

        return y_pred
    
    def confusion_matrix(self, y_true, y_pred):
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        for true, pred in zip(y_true, y_pred):
            matrix[label_to_index[true], label_to_index[pred]] += 1

        return matrix
