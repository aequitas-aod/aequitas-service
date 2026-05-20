import requests
from library.use_cases.tabular.src.local_platform.utils import *
from library.src.artifact_types import Data, Configuration, Report

import numpy as np
import joblib
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix

import holisticai
from holisticai.bias.metrics import (
    disparate_impact,
    statistical_parity,
    average_odds_diff,
)
from holisticai.bias.mitigation import EqualizedOdds
from tools_catalog.code_snippets.alibi_detect import X_test


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

def post_processing_fairness(prediction):
    pass
