from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def train_knn(X_train, y_train, k, weights, metric):
    model = KNeighborsClassifier(
        n_neighbors=k,
        weights=weights,
        metric=metric
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return acc, pd.DataFrame(cm)
