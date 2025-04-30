from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as OvR
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.manifold import TSNE

from OvRClassifier import OvRClassifier
from quality_measures import accuracy, precision, recall, f1_score
from decision_boundaries import plot_decision_boundary

data = load_wine()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

ovr = OvRClassifier(learning_rate=0.01, n_iter=50)
ovr.fit(X_train, y_train)
y_pred = ovr.predict(X_test)

# s = SVC(kernel="linear")
# s.fit(X_train, y_train)
# g = s.predict(X_test)
# print(g)

ovr2 = OvR(SVC(kernel="linear"))
ovr2.fit(X_train, y_train)
y_pred_2 = ovr2.predict(X_test)

print("============ OvR Perceptron ============")
print("Macierz pomyłek:\n", confusion_matrix(y_test, y_pred))
print("\t  Micro    Macro")
print(f"Accuracy  {accuracy(y_test, y_pred) * 100:.2f}%  -")
print(f"Precision {precision(y_test, y_pred, average='micro') * 100:.2f}%  {precision(y_test, y_pred, average='micro') * 100:.2f}%")
print(f"Recall    {recall(y_test, y_pred, average='micro') * 100:.2f}%  {recall(y_test, y_pred, average='micro') * 100:.2f}%")
print(f"F1-score  {f1_score(y_test, y_pred, average='micro') * 100:.2f}%  {f1_score(y_test, y_pred, average='micro') * 100:.2f}%")
# print("\nRaport klasyfikacji:\n", classification_report(y_test, y_pred, target_names=data.target_names, zero_division=0))

print("=============== OvR SVC ===============")
print("Macierz pomyłek:\n", confusion_matrix(y_test, y_pred_2))
print("\t  Micro    Macro")
print(f"Accuracy  {accuracy(y_test, y_pred_2) * 100:.2f}%  -")
print(f"Precision {precision(y_test, y_pred_2, average='micro') * 100:.2f}%  {precision(y_test, y_pred_2, average='micro') * 100:.2f}%")
print(f"Recall    {recall(y_test, y_pred_2, average='micro') * 100:.2f}%  {recall(y_test, y_pred_2, average='micro') * 100:.2f}%")
print(f"F1-score  {f1_score(y_test, y_pred_2, average='micro') * 100:.2f}%  {f1_score(y_test, y_pred_2, average='micro') * 100:.2f}%")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)

X_train_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_train)
X_test_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_test)

ovr_percep = OvRClassifier(learning_rate=0.1, n_iter=100)
ovr_percep.fit(X_train_embedded, y_train)

plot_decision_boundary(ovr_percep, X_embedded, y, "Decision boundaries - OvR Perceptron")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) #balance negative/positive classes

X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)

X_train_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_train)
X_test_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_test)

ovr_svc = OvR(SVC(kernel="linear"))
ovr_svc.fit(X_embedded, y)

plot_decision_boundary(ovr_svc, X_embedded, y, "Decision boundaries - OvR SVC")