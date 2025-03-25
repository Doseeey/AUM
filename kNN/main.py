from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from KNNClassifier import KNNClassifier
from quality_measures import accuracy, precision, recall, f1_score
from sklearn.neighbors import KNeighborsClassifier


def optimize_hiperparams(X, y, k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20], metric_values = [1, 1.5, 2, 3, 4]):
    best_k, best_p, best_score = 0, 0, 0
    for k in k_values:
        for p in metric_values:
            #knn = KNeighborsClassifier(n_neighbors=k, p=p)
            knn = KNNClassifier(k=k, metric=p)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                scores.append(accuracy(y_test, y_pred))

            mean_score = np.array(scores).mean()
            if mean_score > best_score:
                best_k, best_p, best_score = k, p, mean_score
    
    return best_k, best_p, best_score

def tsne(X, y):
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)

    df_tsne = pd.DataFrame(X_embedded, columns=['X', 'Y'])
    df_tsne['target'] = y

    colors = {0:'r', 1:'g', 2:'b'}

    for class_label in np.unique(y):
        plt.scatter(df_tsne[df_tsne['target'] == class_label]['X'],
                    df_tsne[df_tsne['target'] == class_label]['Y'],
                    label=f'Klasa {class_label} - Zbior', marker='o', edgecolors=[colors[class_label]], facecolors='none')
        
    X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.3, random_state=42)

    knn = KNNClassifier(k = 1, metric = 1)
    knn.fit(X_train, y_train) 
    
    y_pred = knn.predict(pd.DataFrame(X_test))

    df_tsne = pd.DataFrame(X_test, columns=['X', 'Y'])
    df_tsne['target'] = y_pred

    for class_label in np.unique(y):
        plt.scatter(df_tsne[df_tsne['target'] == class_label]['X'],
                    df_tsne[df_tsne['target'] == class_label]['Y'],
                    label=f'Klasa {class_label} - Predykcja', marker='o', c=[colors[class_label]], s=15) 

    plt.legend()
    plt.title('Wizualizacja Wine Dataset tSNE')
    plt.show()

def main():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(X_train)

    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(f"Wine Dataset dla kNN (k=3, p=2)")

    print(f"Macierz pomyłek:\n{knn.confusion_matrix(y_test.values, y_pred)}")

    print(f"Accuracy: {accuracy(y_test.values, y_pred)}")
    print(f"Precision: {precision(y_test.values, y_pred)}")
    print(f"Recall: {recall(y_test.values, y_pred)}")
    print(f"F1 score: {f1_score(y_test.values, y_pred)}")

    best_params = optimize_hiperparams(X, y)
    print(f"Optymalne parametry:\n  k: {best_params[0]}\n  p: {best_params[1]}\n  acc: {best_params[2]}")

    tsne(X, y)

main()