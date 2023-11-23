import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import feature_selection
import model_training
import hyperparameter_tuning
import model_evaluation
import joblib
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
    hyperparameter,
    num_features,
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
    data["class_integers"] = class_labels.map(label_mapping)

    if not position_separate:
        data = pd.get_dummies(data, columns=["position"], drop_first=True)

    positions = data["position"].unique() if position_separate else [None]

    for position in positions:
        data_subset = data[data["position"] == position] if position else data

        if position_separate:
            data_subset = data_subset.drop(columns=["position"], axis=1)

        X = data_subset.drop(
            columns=columns_to_drop + [target_column, "class_integers"], axis=1
        )
        y = data_subset["class_integers"]

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
            ) = feature_selection.select_features_random_forest(
                X_train, X_test, y_train, num_features=num_features
            )
        elif feature_selection_model == "rfe":
            X_train_selected, X_test_selected = feature_selection.select_features_rfe(
                X_train, X_test, y_train, num_features=num_features
            )

        # Model Training based on selected type

        if hyperparameter:
            if train_model_type == "xgboost":
                model = hyperparameter_tuning.tune_xgboost(X_train_selected, y_train)
            elif train_model_type == "svm":
                model = hyperparameter_tuning.tune_svm(X_train_selected, y_train)
        else:
            if train_model_type == "xgboost":
                model = model_training.train_xgboost(X_train_selected, y_train)
            elif train_model_type == "svm":
                model = model_training.train_svm(X_train_selected, y_train)

        y_pred = model.predict(X_test_selected)
        model_evaluation.evaluate_model(
            y_test, y_pred, label_mapping, position=position
        )
        model_evaluation.plot_confusion_matrix(
            y_test,
            y_pred,
            label_mapping,
            title=f"Confusion Matrix for {position if position else 'all'}",
            save_path=f"confusion_matrix_{position if position else 'all'}.png",
        )

        if args.train_model == "xgboost":
            model_evaluation.plot_feature_importance(
                model.feature_importances_,
                X.columns,
                title=f"Feature Importance for {position if position else 'all'}",
                save_path=f"feature_importance_{position if position else 'all'}.png",
            )

        model_name = f"{train_model_type}_model_{position if position else 'all'}.pkl"
        model_path = os.path.join(".\models", model_name)
        joblib.dump(model, model_path)
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
        "--hyperparameter",
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

    args = parser.parse_args()

    main(
        args.position_separate,
        args.feature_selection_model,
        args.train_model,
        args.hyperparameter,
        args.num_features,
    )
