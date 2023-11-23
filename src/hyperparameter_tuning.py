from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.svm import SVC


def tune_xgboost(X, y):
    param_grid = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [100, 200, 300],
        "subsample": [0.6, 0.8, 1.0],
    }
    model = xgb.XGBClassifier(use_label_encoder=False)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def tune_svm(X, y):
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": [0.01, 0.1, 1],
        "kernel": ["rbf", "linear"],
    }
    model = SVC()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
    grid_search.fit(X, y)
    return grid_search.best_estimator_
