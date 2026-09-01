import yaml
import requests
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from holisticai.bias.mitigation import EqualizedOdds

from temlops.use_cases.tabular.src.local_platform.utils import split_data_from_df, get_metrics_classifier
from temlops.src.artifact_types import Data, Configuration, Model, Report
from temlops.use_cases.recruitment.src.local_platform.platform_artifacts import (
    DataTabular,
    ReportTabular,
    ModelTabular,
    DocumentationTabular,
)               

import mlflow.sklearn
from mlserver import MLModel, types
from mlserver.utils import get_model_uri
from mlserver.codecs import NumpyCodec


################################## Model deployment #################################


    
################################################################################# Data drift detection

def data_drift_detection_evidently():
    pass


def data_drift_detection_nannyml():
    pass

################################################################################ Model monitoring

def quantify_concept_drift_impact_on_performance():
    pass


def measure_magnitude_of_concept_drift():   
    pass




################################################################################ proxy for inference service


def pre_inference_transformation():
    pass


def post_inference_transformation():
    pass

def post_processing_fairness_validation(data_test: Data, config: Configuration, model: Model, report: Report) -> Report:
    model_test = ModelTabular(model.__dict__).load_model()
    dataset_test = DataTabular(data_test.__dict__).load_dataset()
    report = ReportTabular(report.__dict__)
    report = ReportTabular(report.__dict__)
    report.load_report()
    # Split Testing set to have a post-processor 'Training'
    data_pp_train, data_pp_test = train_test_split(
        dataset_test, test_size=config.test_size, random_state=config.random_state
    )
    X_pp_train, y_pp_train, dem_pp_train = split_data_from_df(data_pp_train)
    X_pp_test, y_pp_test, dem_pp_test = split_data_from_df(data_pp_test)

    group_a_pp_train = dem_pp_train[config.sensitive] == config.group_a
    group_b_pp_train = dem_pp_train[config.sensitive] == config.group_b
    group_a_pp_test = dem_pp_test[config.sensitive] == config.group_a
    group_b_pp_test = dem_pp_test[config.sensitive] == config.group_b

    # Fit processor on the 'training' data
    eq_postprocessor = EqualizedOdds(solver="highs", seed=42)
    fit_params = {
        "group_a": group_a_pp_train,
        "group_b": group_b_pp_train,
    }
    y_pred_pp_train = model.predict(X_pp_train)

    eq_postprocessor.fit(y_pp_train, y_pred_pp_train, **fit_params)
    # ── Log everything to MLflow ─────────────────────────────────────────────────
    with mlflow.start_run(run_name="clf_with_equalized_odds") as run:
        # Save and log the fitted postprocessor as artifact
        with open("postprocessor.pkl", "wb") as f:
            pickle.dump(eq_postprocessor, f)
        mlflow.log_artifact("postprocessor.pkl", artifact_path="postprocessor")
        mlflow.log_params({
            "model":       "LogisticRegression",
            "mitigation":  "EqualizedOdds",
            "constraints": "equalized_odds"
        })

    # Apply Processor to Predictions from 'Test' Data
    fit_params = {
        "group_a": group_a_pp_test,  # Define the first group (e.g., 'Black' candidates)
        "group_b": group_b_pp_test,  # Define the second group (e.g., 'White' candidates)
    }
    y_pred_pp_test = model.predict(X_pp_test)  # Predict the labels for the test set
    d = eq_postprocessor.transform(
        y_pred_pp_test, **fit_params
    )  # Apply equalized odds processor to the predictions

    # Extract the new predictions after applying the fairness processor
    y_pred_pp_new = d["y_pred"]

    # Evaluate and Plot Metrics
    metrics_eq = get_metrics_classifier(
        group_a_pp_test, group_b_pp_test, y_pred_pp_new, y_pp_test, "Black vs White"
    )  # Get fairness metrics
    report.save_report_dataframe(metrics_eq)  # Display the fairness metrics
    return report


def post_processing_bias_mitigation(prediction):
    pass


class FairClassifierRuntime(MLModel):
    """
    Custom MLServer runtime that:
    1. Loads the base classifier from MLflow
    2. Loads the fitted EqualizedOdds postprocessor from MLflow
    3. On every predict() call:
       a. Runs raw model inference
       b. Applies EqualizedOdds debiasing
       c. Returns the debiased prediction
    """

    async def load(self) -> bool:
        """
        Called once at startup. Loads both the classifier
        and the EqualizedOdds postprocessor from MLflow.
        """
        # ── Load base classifier from MLflow ────────────────────────────────
        model_uri = self._settings.parameters.uri
        self._classifier = mlflow.sklearn.load_model(model_uri)

        # ── Load postprocessor artifact from MLflow ──────────────────────────
        # Build the artifact URI relative to the model URI
        run_uri = model_uri.rsplit("/classifier", 1)[0]
        postprocessor_uri = f"{run_uri}/postprocessor/postprocessor.pkl"

        local_path = mlflow.artifacts.download_artifacts(postprocessor_uri)
        with open(local_path, "rb") as f:
            self._postprocessor = pickle.load(f)

        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        """
        Called on every inference request.

        Expects payload inputs:
          - "features":          shape (n_samples, n_features)
          - "sensitive_features": shape (n_samples,)

        Returns:
          - "predictions": debiased class predictions
        """
        # ── Decode inputs ────────────────────────────────────────────────────
        features = NumpyCodec.decode_input(payload.inputs[0])           # (n, features)
        sensitive = NumpyCodec.decode_input(payload.inputs[1])

        # ── Step 1: Raw model inference ──────────────────────────────────────
        raw_predictions = self._classifier.predict(features)

        # ── Step 2: Apply EqualizedOdds post-processing ──────────────────────
        # This is where bias mitigation happens on EVERY request
        # Apply Processor to Predictions from 'production' Data
        group_a_pp_test = sensitive == 0
        group_b_pp_test = sensitive == 1
        fit_params = {
            "group_a": group_a_pp_test,  # Define the first group (e.g., 'Black' candidates)
            "group_b": group_b_pp_test,  # Define the second group (e.g., 'White' candidates)
        }

        debiased_predictions = self._postprocessor.transform(
            raw_predictions, **fit_params
        )  # Apply equalized odds processor to the predictions
        # Extract the new predictions after applying the fairness processor
        debiased_predictions = debiased_predictions["y_pred"]      

        # ── Step 3: Return debiased predictions ─────────────────────────────
        return types.InferenceResponse(
            id=payload.id,
            model_name=self.name,
            outputs=[
                types.ResponseOutput(
                    name="predictions",
                    shape=list(debiased_predictions.shape),
                    datatype="INT64",
                    data=debiased_predictions.tolist(),
                )
            ]
        )

if __name__ == "__main__":
    def _resolve_vars(specs_list, data_artifacts, config_artifacts, model_artifacts):
        vars = {}
        for item in specs_list:
            artifact_name = list(item.values())[0]
            key = list(item.keys())[0]
            match = next((a for a in data_artifacts if a["name"] == artifact_name), None)
            if match:
                vars[key] = Data(**{k: v for k, v in match.items() if k != "name"})
            match = next((a for a in config_artifacts if a["name"] == artifact_name), None)
            if match:
                vars[key] = Configuration(**{k: v for k, v in match.items() if k != "name"})
            match = next((a for a in model_artifacts if a["name"] == artifact_name), None)
            if match:
                vars[key] = Model(**{k: v for k, v in match.items() if k != "name"})
        return vars

    def run_operation(
        operation,
        data_artifacts,
        model_artifacts,
        config_artifacts
    ):
        specs = operation["implementation"]["spec"]
        method_name = specs["method_name"]

        input_vars = _resolve_vars(specs["inputs"], data_artifacts, config_artifacts, model_artifacts)
        input_vars.update(_resolve_vars(specs["outputs"], data_artifacts, config_artifacts, model_artifacts))
        print(input_vars)

        func = globals()[method_name]
        func(**input_vars)
    
    def call_endpoint():
        # Sample input: 5 rows, 10 features + sensitive attribute
        features   = np.random.randn(5, 10).tolist()
        sensitive  = [0, 1, 0, 1, 0]
        payload = {
            "inputs": [
                {
                    "name":     "features",
                    "shape":    [5, 10],
                    "datatype": "FP64",
                    "data":     features
                },
                {
                    "name":     "sensitive_features",
                    "shape":    [5],
                    "datatype": "INT64",
                    "data":     sensitive
                }
            ]
        }
        response = requests.post(
            "http://localhost:8080/v2/models/fair-classifier/infer",
            json=payload
        )
        print(response.json())
     
    with open(
        "../../metadata/aipc_local.yaml",
        "r",
    ) as f:
        aipc_config = yaml.safe_load(f)
    operation = list(
        filter(
            lambda x: x["id"] == "split_train_valid_test_data",
            aipc_config["operations"],
        )
    )[0]
    data_artifacts = aipc_config["artifacts"]["data"]
    model_artifacts = aipc_config["artifacts"]["model"]
    config_artifacts = aipc_config["artifacts"]["configuration"]
    run_operation(
        operation,
        data_artifacts,
        model_artifacts,
        config_artifacts
    )