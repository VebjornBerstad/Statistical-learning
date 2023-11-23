import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


def select_features_random_forest(X_train, X_test, y, num_features):
    model = RandomForestClassifier()
    model.fit(X_train, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    X_train_reduced = X_train[:, indices[:num_features]]
    X_test_reduced = X_test[:, indices[:num_features]]
    return X_train_reduced, X_test_reduced


def select_features_rfe(X_train, X_test, y, num_features):
    model = SVC(kernel="linear")
    selector = RFE(model, n_features_to_select=num_features)
    selector = selector.fit(X_train, y)
    X_rfe = selector.transform(X_train)
    X_test_rfe = selector.transform(X_test)
    return X_rfe, X_test_rfe
