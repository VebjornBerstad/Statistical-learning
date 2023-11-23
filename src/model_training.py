import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def train_xgboost(X, y):
    model = xgb.XGBClassifier()
    model.fit(X, y)
    return model


def train_svm(X, y):
    model = SVC(kernel="rbf")
    model.fit(X, y)
    return model
