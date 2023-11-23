import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import src.feature_selection
import src.model_training
import src.hyperparameter_tuning
import src.model_evaluation
import pickle
import logging
from argparse import ArgumentParser

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(
    position_separate,
    feature_selection_model,
    train_model_type,
    hyper_tuning,
    num_features,
    baseline,
):
    data_path = "data/processed/processed_data.csv"
    data = pd.read_csv(data_path)

    target_column = "total_points"
    columns_to_drop = ["name"]

    class_labels = pd.qcut(
        data[target_column], 4, labels=["low", "medium", "high", "very high"]
    )
    label_mapping = {
        label: idx for idx, label in enumerate(["low", "medium", "high", "very high"])
    }
    data["point_quantiles"] = class_labels.map(label_mapping)

    if baseline:
        baseline_preds = src.model_training.baseline_model(data)
        src.model_evaluation.evaluate_model(
            data["point_quantiles"], baseline_preds, label_mapping
        )
        src.model_evaluation.plot_confusion_matrix(
            data["point_quantiles"],
            baseline_preds,
            label_mapping,
            title="Confusion Matrix for Baseline",
            save_path="confusion_matrix_baseline.png",
        )
        return

    if not position_separate:
        data = pd.get_dummies(data, columns=["position"], drop_first=True)

    positions = data["position"].unique() if position_separate else [None]

    for position in positions:
        data_subset = data[data["position"] == position] if position else data

        if position_separate:
            data_subset = data_subset.drop(columns=["position"], axis=1)

        X = data_subset.drop(
            columns=columns_to_drop + [target_column, "point_quantiles"], axis=1
        )
        y = data_subset["point_quantiles"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Feature Selection based on selected model type
        if feature_selection_model == "randf":
            (
                X_train_selected,
                X_test_selected,
            ) = src.feature_selection.select_features_random_forest(
                X_train, X_test, y_train, num_features=num_features
            )
        elif feature_selection_model == "rfe":
            (
                X_train_selected,
                X_test_selected,
            ) = src.feature_selection.select_features_rfe(
                X_train, X_test, y_train, num_features=num_features
            )

        # Model Training based on selected type

        if hyper_tuning:
            if train_model_type == "xgboost":
                model = src.hyperparameter_tuning.tune_xgboost(
                    X_train_selected, y_train
                )
            elif train_model_type == "svm":
                model = src.hyperparameter_tuning.tune_svm(X_train_selected, y_train)
        else:
            if train_model_type == "xgboost":
                model = src.model_training.train_xgboost(X_train_selected, y_train)
            elif train_model_type == "svm":
                model = src.model_training.train_svm(X_train_selected, y_train)

        y_pred = model.predict(X_test_selected)
        src.model_evaluation.evaluate_model(
            y_test, y_pred, label_mapping, position=position
        )
        src.model_evaluation.plot_confusion_matrix(
            y_test,
            y_pred,
            label_mapping,
            title=f"Confusion Matrix for {position if position else 'all'}",
            save_path=f"confusion_matrix_{position if position else 'all'}.png",
        )

        if args.train_model == "xgboost":
            src.model_evaluation.plot_feature_importance(
                model.feature_importances_,
                X.columns,
                title=f"Feature Importance for {position if position else 'all'}",
                save_path=f"feature_importance_{position if position else 'all'}.png",
            )

        model_name = f"{train_model_type}_model_{feature_selection_model}_{num_features}_feature_selection_{position if position else 'all'}_position.pkl"
        model_path = os.path.join(".\models", model_name)
        pickle.dump(model, open(model_path, "wb"))
        logging.info(f"Model saved as {model_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--position_separate",
        default=False,
        action="store_true",
        help="Create separate models for each position",
    )
    parser.add_argument(
        "--train_model",
        type=str,
        choices=["xgboost", "svm"],
        default="xgboost",
        help="Choose the model for training",
    )
    parser.add_argument(
        "--feature_selection_model",
        type=str,
        choices=["randf", "rfe"],
        default="rfe",
        help="Choose the model for feature selection",
    )
    parser.add_argument(
        "--hyper_tuning",
        default=False,
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=30,
        help="Number of features to select for feature selection",
    )
    parser.add_argument(
        "--baseline", default=False, action="store_true", help="Run baseline model"
    )

    args = parser.parse_args()

    main(
        args.position_separate,
        args.feature_selection_model,
        args.train_model,
        args.hyper_tuning,
        args.num_features,
        args.baseline,
    )
