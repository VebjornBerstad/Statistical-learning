import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def train_xgboost(X, y):
    model = xgb.XGBClassifier()
    model.fit(X, y)
    return model


def train_svm(X, y, kernel="rbf"):
    model = SVC(kernel=kernel)
    model.fit(X, y)
    return model


def baseline_model(
    data, minutes_played_column="Min", point_quantiles_column="point_quantiles"
):
    min_quantiles = pd.qcut(data[minutes_played_column], 4, labels=False)
    data["baseline_pred"] = min_quantiles

    return data["baseline_pred"]
