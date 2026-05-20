import os
import json
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Any, Tuple
from library.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation
import DataTabular, ReportTabular


""" 
Data preparation stage containing 4 operations
Data Profiling
Data Validation
Data Preprocessing
Data Documentation
"""

################################################################################################## Data Preprocessing

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
REPORTS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")

def load_data(data: Data) -> Data:    
    data = DataTabular(data.__dict__)
    print(data.__dict__)
    dataset = data.load_dataset()
    print(dataset)
    dataset.drop("Id", axis=1, inplace=True)
    data.log_dataset(dataset)
    return data


def data_validation_check_quantity(data: Data, config: Configuration, output_status: Status) -> Status:
    pass


def split_train_valid_test_data(data, test_size, valid_size, random_state):
    pass


def preprocess_train_data(data_input: Data, data_output: Data):
    pass

def data_validation_demographics_qty(data: Data, config: Configuration, output_status: Status) -> Status:
    pass


def preprocess_reweighing(data_input: Data, data_output: Data):
    pass


def data_card_generation(data: Data, documentation: Documentation):
    pass

def data_profiling_custom(
    data: Data, 
    config: Configuration
    ):
    data = DataTabular(data.__dict__)
    df = data.load_dataset()
    pass


def data_drift_detection(data: Data, config: Report):
    pass


def data_drift_status(data: Data, output_status: Status):
    pass

