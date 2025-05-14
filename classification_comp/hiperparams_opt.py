from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from quality_measures import accuracy

import numpy as np

data = load_wine()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

X_test, X_val, y_test, y_val = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest)

#Bagging + GridSearch

bagging = BaggingClassifier(random_state=1)

bagging_params = {
    "n_estimators": [10, 20, 30, 50, 75, 100],
    "max_samples": [0.1, 0.3, 0.5, 0.8, 1.0]
}

grid_bagging = GridSearchCV(bagging, bagging_params, cv=5, n_jobs=-1)
grid_bagging.fit(X_train, y_train)

best_bagging = BaggingClassifier(random_state=1, 
                                 n_estimators=grid_bagging.best_params_["n_estimators"],
                                 max_samples=grid_bagging.best_params_["max_samples"]
                                 )

best_bagging.fit(X_train, y_train)
y_pred = best_bagging.predict(X_val)

print(f"Bagging best params: {grid_bagging.best_params_}\nAccuracy: {accuracy(y_val, y_pred):.2f}")

adaboost = AdaBoostClassifier(random_state=3)

adaboost_params = {
    "n_estimators": [10, 20, 30, 50, 75, 100],
    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
}

grid_ada = GridSearchCV(adaboost, adaboost_params, cv=5, n_jobs=-1)
grid_ada.fit(X_train, y_train)

best_ada = AdaBoostClassifier(random_state=3,
                              n_estimators=grid_ada.best_params_["n_estimators"],
                              learning_rate=grid_ada.best_params_["learning_rate"]
                              )

best_ada.fit(X_train, y_train)
y_pred = best_ada.predict(X_val)

print(f"Ada best params: {grid_bagging.best_params_}\nAccuracy: {accuracy(y_val, y_pred):.2f}")