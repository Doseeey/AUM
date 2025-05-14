from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

from KNNClassifier import KNNClassifier
from OvRClassifier import OvRClassifier

from quality_measures import accuracy, precision, recall, f1_score

data = load_wine()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#kNN best params - k=1 metric=1
#ovr percep best params - learning_rate=0.11
#randomforest - bez optymalizacji
#bagging - n_estimators=20 max_samples=0.1
#adaboost - n_estimators=20 learning_rate=0.1

models = {
    "knn": KNNClassifier(k=1, metric=1),
    "perceptron_ovr": OvRClassifier(learning_rate=0.11, n_iter=100),
    "randomforest": RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1),
    "bagging": BaggingClassifier(n_estimators=20, max_samples=0.1, random_state=2),
    "adaboost": AdaBoostClassifier(n_estimators=20, learning_rate=0.1, random_state=3)
}

for name, estimator in models.items():

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print("==========================")
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print("           Micro   Macro")
    print(f"Precision: {precision(y_test, y_pred, average='micro'):.4f}  {precision(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall:    {recall(y_test, y_pred, average='micro'):.4f}  {recall(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, average='micro'):.4f}  {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(confusion_matrix(y_test, y_pred))

    
