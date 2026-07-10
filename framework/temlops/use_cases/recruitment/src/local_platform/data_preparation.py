import sys
sys.path.append("../../../../")

import os
import yaml
import openml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Any, Tuple
from temlops.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation
from temlops.use_cases.recruitment.src.local_platform.platform_artifacts import DataTabular, ReportTabular

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlib import DataFrame
from fairlib.preprocessing import Reweighing, DisparateImpactRemover, LFR

""" 
Data preparation stage containing 4 operations categories:
Data Profiling
Data Validation
Data Preprocessing
Data Documentation
"""

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
REPORTS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")

########################################################### Data Profiling

def load_data(data: Data, data_processed: Data) -> Data:
    adult_ds = openml.datasets.get_dataset(data.filepath)
    adult_df, *_ = adult_ds.get_data(dataset_format="dataframe")

    adult_df.rename(columns={"class": "income"}, inplace=True)
    adult_df.drop(columns=["fnlwgt"], inplace=True)
    
    DataTabular(data_processed.__dict__).log_dataset(adult_df)
    return data_processed


def data_profiling(data: Data, report: Report) -> Report:
    data = DataTabular(data.__dict__)
    dataset = data.load_dataset()
    parameters={
        "class_attribute":{
            "name": 'Status',
        },
    }
    Aeq_dataset=Aequitas(dataset,parameters)
    result = Aeq_dataset.descriptive_stats(verbose=True)
    report.save_report_stats(result)
    return report
    
def data_profiling_custom(
    data: Data, 
    config: Configuration
    ):
    data = DataTabular(data.__dict__)
    df = data.load_dataset()
    pass

########################################################### Data Validation
    
def data_validation_check_quantity(data: Data, config: Configuration, output_status: Status) -> Status:
    pass

def data_validation_demographics_qty(data: Data, config: Configuration, output_status: Status) -> Status:
    pass

def data_drift_detection(data: Data, config: Report):
    pass

def data_drift_status(data: Data, output_status: Status):
    pass



########################################################### Data Preprocessing

def split_train_valid_test_data(data: Data, config: Configuration, data_train: Data, data_test: Data, data_valid: Data) -> Tuple[Data, Data, Data]:
    dataset = DataTabular(data.__dict__).load_dataset()
    for col in dataset.columns:
        if dataset[col].dtype == "object" or dataset[col].dtype.name == "category":
            dataset[col], _ = pd.factorize(dataset[col])
    # First split: train+val vs test
    X_train_val, X_test = train_test_split(
        dataset, test_size=config.test_size, random_state=config.random_state
    )
    # Second split: train vs validation
    X_train, X_val = train_test_split(
        X_train_val, test_size=config.valid_size, random_state=config.random_state  # 0.25 x 0.8 = 0.2
    )
    DataTabular(data_train.__dict__).log_dataset(X_train)
    DataTabular(data_test.__dict__).log_dataset(X_test)
    DataTabular(data_valid.__dict__).log_dataset(X_val)
        
    return data_train, data_test, data_valid


def preprocess_train_data(data_input: Data, data_output: Data):
    pass


def preprocess_reweighing(data_input: Data, data_output: Data):
    pass

########################################################### Data Documentation

def data_card_generation(data: Data, documentation: Documentation):
    pass





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