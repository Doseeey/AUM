from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from KNNClassifier import KNNClassifier
from quality_measures import accuracy, precision, recall, f1_score


def optimize_k(X, y, k_from, k_to):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    errors = []
    k_values = range(k_from, k_to + 1)

    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        errors.append(1 - accuracy(y_test, y_pred))

    print(f"Najlepsze k: {k_values[np.argmin(errors)]}\nAccuracy: {accuracy(y_test, y_pred)}")

    plt.plot(k_values, errors, marker='o')
    plt.xlabel("Liczba sąsiadów (k)")
    plt.ylabel("Błąd klasyfikacji (1 - accuracy)")
    plt.title("Wykres błędu klasyfikacji dla różnych k")
    plt.show()

def optimize_p(X, y, k, metrics = [1, 1.5, 2, 3, 4]):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    acc = []

    for p in metrics:
        knn = KNNClassifier(k=k, metric=p)
        knn.fit(X_train, y_train) 
        y_pred = knn.predict(X_test)

        acc.append(accuracy(y_test, y_pred))

    print(f"Najlepsze p: {metrics[np.argmax(acc)]}\nAccuracy: {accuracy(y_test, y_pred)}")

    plt.plot(metrics, acc, marker='o')
    plt.xlabel("Wartość p w metryce Minkowskiego")
    plt.ylabel("Accuracy")
    plt.title("Porównanie accuracy dla różnych metryk")
    plt.show()

def tsne(X, y):
    X_embedded = TSNE(n_components=2).fit_transform(X)

    df_tsne = pd.DataFrame(X_embedded, columns=['X', 'Y'])
    df_tsne['target'] = y

    for class_label in np.unique(y):
        plt.scatter(df_tsne[df_tsne['target'] == class_label]['X'],
                    df_tsne[df_tsne['target'] == class_label]['Y'],
                    label=f'Klasa {class_label}')

    plt.legend()
    plt.title('Wizualizacja Wine Dataset tSNE')
    plt.show()

def tsne_params(X, y, k_val = [1, 3, 5], p_val = [1, 2]):
    _, axs = plt.subplots(len(p_val), len(k_val))
    _.suptitle("tSNE dla różnych k i p")
    _.set_figheight(5)
    _.set_figwidth(10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_embedded_test = TSNE(n_components=2).fit_transform(X_test)

    for i, k in enumerate(k_val):
        for j, p in enumerate(p_val):
            knn = KNNClassifier(k = k, metric = p)
            knn.fit(X_train, y_train) 
            y_pred = knn.predict(X_test)

            df_tsne = pd.DataFrame(X_embedded_test, columns=['X', 'Y'])
            df_tsne['target'] = y_pred

            for class_label in np.unique(y):
                axs[j, i].scatter(df_tsne[df_tsne['target'] == class_label]['X'],
                                  df_tsne[df_tsne['target'] == class_label]['Y'],
                                  label=f'Klasa {class_label}') 
            
            axs[j, i].set_title(f"tSNE dla KNN(k={k},p={p})")
            axs[j, i].legend()

    plt.show()

def main():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #print(X_train)

    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(f"Wine Dataset dla kNN (k=3, p=2)")

    print(f"Macierz pomyłek:\n{knn.confusion_matrix(y_test.values, y_pred)}")

    print(f"Accuracy: {accuracy(y_test.values, y_pred)}")
    print(f"Precision: {precision(y_test.values, y_pred)}")
    print(f"Recall: {recall(y_test.values, y_pred)}")
    print(f"F1 score: {f1_score(y_test.values, y_pred)}")

    print(f"\nOptymalizacja parametru k:\n")
    optimize_k(X, y, k_from=1, k_to=30)

    print(f"Optymalizacja metryki:\n")
    optimize_p(X, y, k=3)

    tsne(X, y)

    tsne_params(X, y)

main()