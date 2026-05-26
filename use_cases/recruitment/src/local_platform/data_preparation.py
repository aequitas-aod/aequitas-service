import sys
sys.path.append("../../../../")

import os
import yaml
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Any, Tuple
from library.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation
from .platform_artifacts import DataTabular, ReportTabular

from sklearn.linear_model import LogisticRegression
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

def load_data(data: Data) -> Data:    
    data = DataTabular(data.__dict__)
    dataset = data.load_dataset()
    #dataset.drop("Id", axis=1, inplace=True)
    data.log_dataset(dataset)
    return data

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

def evaluate_fairness_spd(data:Data, config: Configuration):
    
    data = DataTabular(data.__dict__)
    dataset = data.load_dataset()
    X_test = dataset.drop(config.target, axis=1)
    y_pred = dataset[config.target]
    
    X_eval = X_test.copy()
    X_eval[config.target] = y_pred
    ds_eval = DataFrame(X_eval)
    ds_eval.targets, ds_eval.sensitive = config.target, config.sensitive

    spd = ds_eval.statistical_parity_difference()[{config.target: config.positive_target, config.sensitive: config.favored_class}]
    return spd

def evaluate_fairness_di(data: Data, config: Configuration):
    data = DataTabular(data.__dict__)
    dataset = data.load_dataset()
    X_test = dataset.drop(config.target, axis=1)
    y_pred = dataset[config.target]

    X_eval = X_test.copy()
    X_eval[config.target] = y_pred
    ds_eval = DataFrame(X_eval)
    ds_eval.targets, ds_eval.sensitive = config.target, config.sensitive

    di = ds_eval.disparate_impact()[{config.target: config.positive_target, config.sensitive: config.favored_class}]
    return di

########################################################### Data Preprocessing

def split_train_valid_test_data(data: Data, config: Configuration) -> Tuple[Data, Data]:
    dataset = DataTabular(data.__dict__).load_dataset()
    training_sample, test_sample = dm.split_dataset(dataset, ratio=config.ratio, random_state=config.random_state)
    return training_sample, test_sample


def preprocess_train_data(data_input: Data, data_output: Data):
    pass


def preprocess_reweighing(data_input: Data, data_output: Data):
    pass

########################################################### Data Documentation

def data_card_generation(data: Data, documentation: Documentation):
    pass





if __name__ == "__main__":
    with open(
        "../../metadata/aipc_local.yaml",
        "r",
    ) as f:
        aipc_config = yaml.safe_load(f)
    data_artifact = list(
        filter(
            lambda x: x["name"] == "hiring_data_original",
            aipc_config["artifacts"]["data"],
        )
    )[0]["config"]
    data = Data(data_artifact)
    load_data(data)