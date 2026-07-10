from sklearn.model_selection import train_test_split
from holisticai.bias.mitigation import Reweighing
from holisticai.bias.mitigation import GridSearchReduction
from holisticai.bias.mitigation import EqualizedOdds

# Pre-process Reweighing


def preprocess_bias_mitigation():
    ########## Step1: Initialise and fit the Reweighing model to mitigate bias
    rew = Reweighing()
    # Define the groups (Black and White) in the training data based on the 'Ethnicity' column
    group_a_train = dem_train["Ethnicity"] == "Black"  # Group A: Black ethnicity
    group_b_train = dem_train["Ethnicity"] == "White"  # Group B: White ethnicity

    # Fit the reweighing technique to adjust sample weights
    rew.fit(y_train, group_a_train, group_b_train)

    # Extract the calculated sample weights from the reweighing model
    sample_weights = rew.estimator_params["sample_weight"]
    data_train["sample_weights"] = sample_weights
    # display(data_train.groupby(["Label", "Ethnicity"])["sample_weights"].mean())

    ########## Step2: Train the model using the sample weights calculated through reweighing
    model = RidgeClassifier(random_state=42)
    model.fit(
        X_train, y_train, sample_weight=sample_weights.ravel()
    )  # Fit model with sample weights

    y_pred_test = model.predict(X_test)

    # Define the groupings for fairness analysis (Black and White) in the test set
    group_a_test = dem_test["Ethnicity"] == "Black"
    group_b_test = dem_test["Ethnicity"] == "White"

    # Get the fairness and accuracy metrics after applying reweighing
    metrics_rw = get_metrics(group_a_test, group_b_test, y_pred_test, y_test)
    display(metrics_rw)

    # Add a 'mitigation' column to both metrics dataframes to label them accordingly
    metrics_orig["mitigation"] = "None"
    metrics_rw["mitigation"] = "Reweighing"

    metrics = pd.concat([metrics_orig, metrics_rw], axis=0, ignore_index=True)

    # Plot the comparison of metrics between the original model and the model with reweighing
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics, x="Metric", y="Value", hue="mitigation")
    plt.axhline(y=0.8, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=-0.05, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=1, linewidth=2, color="g")
    plt.axhline(y=0, linewidth=2, color="g")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.show()


def inprocess_mitigation():
    # Train the model using Grid Search Reduction (GSR) bias mitigation technique
    model = RidgeClassifier(random_state=42)

    gsr = GridSearchReduction()
    gsr.transform_estimator(model)

    # Define the groupings for fairness analysis (Black and White) in the training set
    group_a_train = (
        dem_train["Ethnicity"] == "Black"
    )  # Define Black group in training set
    group_b_train = (
        dem_train["Ethnicity"] == "White"
    )  # Define White group in training set

    # Fit GSR with the training data, labels, and group membership
    gsr.fit(X_train, y_train, group_a_train, group_b_train)

    y_pred_test = gsr.predict(X_test)

    # Define groupings for fairness analysis in the test set (Black and White)
    group_a_test = dem_test["Ethnicity"] == "Black"  # Define Black group in test set
    group_b_test = dem_test["Ethnicity"] == "White"  # Define White group in test set

    # Evaluate the fairness and accuracy metrics for the model with GSR mitigation applied
    metrics_gsr = get_metrics(
        group_a_test, group_b_test, y_pred_test, y_test
    )  # Get metrics after gsr

    display(metrics_gsr)

    metrics_orig["mitigation"] = "None"  # Label for the original model (no mitigation)
    metrics_gsr["mitigation"] = "Grid Search Reduction"  # Label for the GSR model

    metrics = pd.concat([metrics_orig, metrics_gsr], axis=0, ignore_index=True)

    # Plot the comparison of metrics between the original model and the model with Grid Search Reduction
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics, x="Metric", y="Value", hue="mitigation")
    plt.axhline(y=0.8, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=-0.05, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=1, linewidth=2, color="g")
    plt.axhline(y=0, linewidth=2, color="g")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.show()


def post_process_mitigation():
    # Split Testing set to have a post-processor 'Training'
    data_pp_train, data_pp_test = train_test_split(
        data_test, test_size=0.4, random_state=42
    )
    X_pp_train, y_pp_train, dem_pp_train = split_data_from_df(data_pp_train)
    X_pp_test, y_pp_test, dem_pp_test = split_data_from_df(data_pp_test)

    group_a_pp_train = dem_pp_train["Ethnicity"] == "Black"
    group_b_pp_train = dem_pp_train["Ethnicity"] == "White"
    group_a_pp_test = dem_pp_test["Ethnicity"] == "Black"
    group_b_pp_test = dem_pp_test["Ethnicity"] == "White"

    # Fit processor on the 'training' data
    eq = EqualizedOdds(solver="highs", seed=42)
    fit_params = {
        "group_a": group_a_pp_train,
        "group_b": group_b_pp_train,
    }
    y_pred_pp_train = model.predict(X_pp_train)

    eq.fit(y_pp_train, y_pred_pp_train, **fit_params)

    # Apply Processor to Predictions from 'Test' Data
    fit_params = {
        "group_a": group_a_pp_test,  # Define the first group (e.g., 'Black' candidates)
        "group_b": group_b_pp_test,  # Define the second group (e.g., 'White' candidates)
    }

    y_pred_pp_test = model.predict(X_pp_test)  # Predict the labels for the test set

    d = eq.transform(
        y_pred_pp_test, **fit_params
    )  # Apply equalized odds processor to the predictions

    # Extract the new predictions after applying the fairness processor
    y_pred_pp_new = d["y_pred"]

    # Evaluate and Plot Metrics
    metrics_eq = get_metrics(
        group_a_pp_test, group_b_pp_test, y_pred_pp_new, y_pp_test
    )  # Get fairness metrics
    display(metrics_eq)  # Display the fairness metrics

    metrics_orig["mitigation"] = "None"
    metrics_eq["mitigation"] = "Equalized Odds"
    metrics = pd.concat([metrics_orig, metrics_eq], axis=0, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics, x="Metric", y="Value", hue="mitigation")
    plt.axhline(y=0.8, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=-0.05, linewidth=2, color="r", linestyle="--")
    plt.axhline(y=1, linewidth=2, color="g")
    plt.axhline(y=0, linewidth=2, color="g")
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.show()
