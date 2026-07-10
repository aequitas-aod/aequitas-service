def plot_confusion_matrices(self, groups, data_test, category, y_test, y_pred_test):
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
