from Perceptron import Perceptron
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from quality_measures import accuracy, precision, recall, f1_score
import matplotlib.pyplot as plt

def opt_params():
    acc = []
    rate = np.arange(0.01, 1.00, 0.02)
    best_acc = 0
    best_rate = 0


    for learning_rate in rate:
        perceptron = Perceptron(learning_rate=learning_rate, n_iter=500)
        perceptron.fit(X_train, y_train)

        y_pred = perceptron.predict(X_test)

        rate_acc = accuracy(y_test, y_pred)
        acc.append(rate_acc)
        print(learning_rate)

        if rate_acc > best_acc:
            best_acc = rate_acc
            best_rate = learning_rate

    return acc, rate, best_acc, best_rate
  
# fetch dataset 
banknote_authentication = fetch_ucirepo(id=267) 
 
X = np.array(banknote_authentication.data.features, dtype=float) 
y = np.array(banknote_authentication.data.targets, dtype=int) 

# print(f"1: {np.count_nonzero(y)} 0: {len(y) - np.count_nonzero(y)}")
# print(X)
# print(X.ravel().max(), X.ravel().min())
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# print(X)
# print(X.ravel().max(), X.ravel().min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

perceptron = Perceptron(learning_rate=0.01, n_iter=1000)
perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

print(f"Accuracy: {accuracy(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision(y_test, y_pred) * 100:.2f}%")
print(f"Recall: {recall(y_test, y_pred) * 100:.2f}%")
print(f"F1 score: {f1_score(y_test, y_pred) * 100:.2f}%")

print(f"Accuracy: {accuracy(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision(y_test, y_pred, average='micro') * 100:.2f}%")
print(f"Recall: {recall(y_test, y_pred, average='micro') * 100:.2f}%")
print(f"F1 score: {f1_score(y_test, y_pred, average='micro') * 100:.2f}%")


acc_range, rate_range, best_acc, best_rate = opt_params()
print(f"Best accuracy: {best_acc * 100:.2f} on learning rate - {best_rate}")

plt.plot(rate_range, acc_range, marker="o")
plt.title("Perceptron accuracy based on learning_rate")
plt.xlabel("learning_rate")
plt.ylabel("accuracy")
plt.show()