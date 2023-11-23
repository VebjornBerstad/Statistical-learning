import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def evaluate_model(y_true, y_pred, label_mapping, position=None):
    # Reverse the label mapping to map integers back to original labels
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # Replace integer labels in y_true and y_pred with original string labels
    y_true_labels = [reverse_label_mapping.get(label, "Unknown") for label in y_true]
    y_pred_labels = [reverse_label_mapping.get(label, "Unknown") for label in y_pred]

    # Sort labels according to the order in label_mapping
    sorted_labels = [reverse_label_mapping[i] for i in sorted(reverse_label_mapping)]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Print metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Print classification report with sorted labels
    print(f"\nClassification Report for {position if position else 'all'}:\n")
    print(
        classification_report(y_true_labels, y_pred_labels, target_names=sorted_labels)
    )


def plot_confusion_matrix(
    y_true,
    y_pred,
    label_mapping,
    title="Confusion Matrix",
    save_path="confusion_matrix.png",
):
    # Reverse the label mapping
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # Replace integer labels with original string labels
    y_true_labels = [reverse_label_mapping.get(label, "Unknown") for label in y_true]
    y_pred_labels = [reverse_label_mapping.get(label, "Unknown") for label in y_pred]

    # Sort labels according to the order in label_mapping
    sorted_labels = [reverse_label_mapping[i] for i in sorted(reverse_label_mapping)]

    # Generate confusion matrix
    matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=sorted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted_labels,
        yticklabels=sorted_labels,
    )
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.savefig("./report/figures/evaluation/" + save_path)
    # plt.show()


def plot_feature_importance(
    importances,
    feature_names,
    title="Feature Importances",
    max_features=10,
    save_path="feature_importance.png",
):
    # Sort the feature importances and select the top ones
    indices = np.argsort(importances)[-max_features:]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(
        {"Feature": sorted_feature_names, "Importance": sorted_importances}
    )

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=df.sort_values(by="Importance", ascending=True),
    )
    plt.title(title)
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.savefig("./report/figures/evaluation/" + save_path)
    # plt.show()
