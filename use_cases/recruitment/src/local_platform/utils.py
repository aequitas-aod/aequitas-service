import pandas as pd
from io import BytesIO
from datetime import datetime

from holisticai.bias.metrics import (
    disparate_impact,
    statistical_parity,
    average_odds_diff,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns


def split_data_from_df(data, sensitive_features):
    """
    Splits a DataFrame into features (X), labels (y), and demographic data (dem).
    """
    filter_col = sensitive_features
    features = data.drop(columns=["Id", "decision"] + filter_col).columns
    y = data["decision"].values  # Extract labels
    X = data[features].values  # Extract features
    dem = data[filter_col].copy()  # Extract demographics
    return X, y, dem  # Return features, labels, demographics


def get_metrics_classifier(group_a, group_b, y_pred, y_true, group_label):
    """
    Function to calculate and return model accuracy and fairness metrics for two groups
    Returns a DataFrame of model accuracy and fairness metrics for two groups.
    """
    metrics = [
        ["Model Accuracy", round(accuracy_score(y_true, y_pred), 2), 1]
    ]  # Calculate accuracy
    metrics += [
        ["Precision", round(precision_score(y_true, y_pred), 2), 1]
    ]  # Calculate precision: Of the predited positives (TP + FP), how many are correctly predicted
    metrics += [
        ["Recall", round(recall_score(y_true, y_pred), 2), 1]
    ]  # Calculate recall: Of the actual positives (TP + FN), how many were correctly predicted
    metrics += [
        ["F1 Score", round(f1_score(y_true, y_pred), 2), 1]
    ]  # Calculate f1-score
    metrics += [
        [
            f"{group_label} Disparate Impact",
            round(disparate_impact(group_a, group_b, y_pred), 2),
            1,
        ]
    ]  # Calculate disparate impact
    metrics += [
        [
            f"{group_label} Statistical Parity",
            round(statistical_parity(group_a, group_b, y_pred), 2),
            0,
        ]
    ]  # Calculate statistical parity
    metrics += [
        [
            f"{group_label} Average Odds Difference",
            round(average_odds_diff(group_a, group_b, y_pred, y_true), 2),
            0,
        ]
    ]  # Calculate average odds difference
    return pd.DataFrame(
        metrics, columns=["Metric", "Value", "Reference"]
    )  # Return metrics as DataFrame


def compare_metrics(metrics):
    now = datetime.now()
    # Plot the comparison of metrics between the original model and the model with reweighing
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics, x="Metric", y="Value", hue="mitigation")
    plt.axhline(y=0.8, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=-0.05, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=1, linewidth=2, color="g")
    plt.axhline(y=0, linewidth=2, color="g")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    # plt.show()
    # Save the figure
    plt.savefig(f"metrics_comparison_{now}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_to_str():
    img = BytesIO()
    plt.savefig(img, format="png")
    return base64.encodebytes(img.getvalue()).decode("utf-8")


# Function to calculate True Positive Rate (TPR) from confusion matrices
def calculate_tpr(cms):
    """
    Calculates True Positive Rates (TPR) for each group,
    given a set of confusion matrices.
    """
    tprs = {g: cm[0, 0] / cm[0, :].sum() for g, cm in cms.items()}  # Calculate TPR
    return tprs  # Return dictionary of TPRs


# Function to plot confusion matrices for different groups in a dataset
def plot_confusion_matrices(groups, data_test, category, y_test, y_pred_test):
    """
    Plots confusion matrices for each group in a given category.
    """
    num_groups = len(groups) + 1  # Number of groups to display
    fig, axes = plt.subplots(
        1, num_groups, figsize=(5 * num_groups, 4)
    )  # Create subplot grid

    # Plot confusion matrix for overall data
    cm = plot_cm(y_test, y_pred_test, ax=axes[0])
    axes[0].set_title("All", fontsize=14, fontweight="bold")

    # Plot confusion matrices for each group in the dataset
    cm_dict = {"All": cm}  # Store overall confusion matrix
    for i, group in enumerate(groups):
        ax = axes[i + 1]  # Get axis for group
        subset = data_test[data_test[category] == group]  # Filter data for group
        cm = plot_cm(
            subset["Label"], subset["Pred"], ax=ax
        )  # Plot confusion matrix for group
        cm_dict[group] = cm  # Store confusion matrix for group
        ax.set_title(group, fontsize=14, fontweight="bold")

    plt.tight_layout()  # Adjust layout
    plt.show()  # Display plot
    return cm_dict  # Return dictionary of confusion matrices for each group


def plot_cm(y_true, y_pred, labels=[1, 0], display_labels=[1, 0], ax=None):
    """
    Plots a single confusion matrix with annotations
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)  # Compute confusion matrix

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(4, 3)
        )  # Create new figure if no axis is provided

    # Create heatmap for confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="viridis",
        cbar=False,
        xticklabels=display_labels,
        yticklabels=display_labels,
        square=True,
        linewidths=2,
        linecolor="black",
        ax=ax,
        annot_kws={"size": 14},
    )

    # Label and format axes
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)

    return cm  # Return confusion matrix


def generate_model_card():
    pass
    # TODO
    # mct = ModelCardToolkit()
    # model_card = mct.scaffold_assets()
