import sys
sys.path.append("../../../../")

import os
import yaml
import pickle
import logging
import pandas as pd

from temlops.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation
from use_cases.recruitment.src.local_platform.platform_artifacts import DataTabular, ReportTabular, ModelTabular

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlib import DataFrame
from fairlib.preprocessing import Reweighing, DisparateImpactRemover, LFR
from fairlib.inprocessing import Fauci, AdversarialDebiasing

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")
REPORT_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
STATUS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "status")


#################################################### Model Training ####################################################



def train_model_baseline(data: Data, config: Configuration, model: Model) -> Model:
    dataset = DataTabular(data.__dict__).load_dataset()
    X_train = dataset.drop(columns=[config.target_column]).copy(deep=True)
    y_train = dataset[config.target_column].copy(deep=True)   
    
    base_clf = train_classifier(X_train, y_train, random_state=config.random_state, max_iter=config.max_iter)
    return ModelTabular(model.__dict__).save_model(base_clf)
    
    
#################################### pre-processing techniques
def train_model_reweighing(data: Data, config: Configuration, model: Model) -> Model:
    dataset = DataTabular(data.__dict__).load_dataset()
    X_train = dataset.drop(columns=[config.target_column]).copy(deep=True)
    y_train = dataset[config.target_column].copy(deep=True)
    
    train_rw = X_train.copy(); train_rw[config.target_column] = y_train
    ds_rw = DataFrame(train_rw); ds_rw.targets, ds_rw.sensitive = config.target_column, config.sensitive
    rw_proc = Reweighing(); ds_rw_t = rw_proc.fit_transform(ds_rw)
    
    rw_clf = train_classifier(X_train, y_train, sample_weight=ds_rw_t["weights"].values)
    return ModelTabular(model.__dict__).save_model(rw_clf)

def train_model_disparate_impact_remover():
    pass

def train_model_learning_fair_representations():
    pass

#################################### in-processing techniques
def train_model_fauci():
    pass

def train_model_adversarial_debiasing():
    pass



############################################################## Model Evaluations - Performance and Fairness metrics


def model_evaluation_performance(data_test: Data, config: Configuration, model: Model, report: Report) -> Report:
    model_test = ModelTabular(model.__dict__).load_model()
    dataset_test = DataTabular(data_test.__dict__).load_dataset() 
    report = ReportTabular(report.__dict__)   
    X_test = dataset_test.drop(columns=[config.target_column]).copy(deep=True)
    y_test = dataset_test[config.target_column].copy(deep=True)
    
    base_pred = model_test.predict(X_test)    
    base_acc = accuracy_score(y_test, base_pred)
    report.save_report_dataframe(pd.DataFrame([{"algorithm": model.filepath.split(".")[0], "accuracy": base_acc}]))
    return report


def model_evaluation_fairness(data_test: Data, config: Configuration, model: Model, report: Report) -> Report:
    data = DataTabular(data_test.__dict__)
    dataset = data.load_dataset()
    X_test = dataset.drop(config.target, axis=1)
    y_pred = dataset[config.target]
    
    X_eval = X_test.copy()
    X_eval[config.target] = y_pred
    ds_eval = DataFrame(X_eval)
    ds_eval.targets, ds_eval.sensitive = config.target, config.sensitive

    spd = ds_eval.statistical_parity_difference()[{config.target: config.positive_target, config.sensitive: config.favored_class}]
    di = ds_eval.disparate_impact()[{config.target: config.positive_target, config.sensitive: config.favored_class}]
    report.save_report_dataframe(pd.DataFrame([{"algorithm": model.filepath.split(".")[0], "spd": spd, "di": di}]))
    return report


def check_all_fairness_metrics():
    pass


############# Model Validation


def model_validation_baseline(report: Report, config: Configuration, status: Status):
    pass


def check_all_fairness_metrics():
    pass


############################################################## In Process Bias Mitigation Techniques


def bias_mitigation_in_process_train(data: Data, config: Configuration, report: Report):
    pass

############################################################### helper functions

def train_classifier(X, y, sample_weight=None, random_state=42, max_iter=1000):
    clf = LogisticRegression(random_state=random_state, max_iter=max_iter)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf



if __name__ == "__main__":
    def _resolve_vars(specs_list, data_artifacts, config_artifacts, model_artifacts, report_artifacts):
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
            match = next((a for a in report_artifacts if a["name"] == artifact_name), None)
            if match:
                vars[key] = Report(**{k: v for k, v in match.items() if k != "name"})
        return vars

    def run_operation(
        operation,
        data_artifacts,
        model_artifacts,
        config_artifacts,
        report_artifacts
    ):
        specs = operation["implementation"]["spec"]
        method_name = specs["method_name"]

        input_vars = _resolve_vars(specs["inputs"], data_artifacts, config_artifacts, model_artifacts, report_artifacts)
        input_vars.update(_resolve_vars(specs["outputs"], data_artifacts, config_artifacts, model_artifacts, report_artifacts))
        print(input_vars)

        func = globals()[method_name]
        func(**input_vars)
       
    with open(
        "../../metadata/aipc_local.yaml",
        "r",
    ) as f:
        aipc_config = yaml.safe_load(f)
    operation = list(
        filter(
            lambda x: x["id"] == "train_model_reweighing",
            aipc_config["operations"],
        )
    )[0]
    data_artifacts = aipc_config["artifacts"]["data"]
    model_artifacts = aipc_config["artifacts"]["model"]
    config_artifacts = aipc_config["artifacts"]["configuration"]
    report_artifacts = aipc_config["artifacts"]["report"]
    run_operation(
        operation,
        data_artifacts,
        model_artifacts,
        config_artifacts,
        report_artifacts
    )