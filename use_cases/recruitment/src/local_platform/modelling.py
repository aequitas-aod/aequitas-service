import os
import pickle
import logging
from pickle import dump


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")
REPORT_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
STATUS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "status")


#################################################### Model Training ####################################################



def train_model(data: Data, config: Configuration):
    """ """
    pass


############################################################## Evaluations - Performance metrics - Accuracy


def model_evaluation_accuracy(
    data: Data, config: Configuration, model: Model, report: Report
):
    model = model.load_model()
    # prepare a validation dataset for prediction and predict
    data_test = data.load_dataset()
    pass


############## Evaluations - Performance and Fairness evaluation metrics


def model_evaluation_accuracy_demographic_groups(
    model: Model, data_valid: Data, config: Configuration
):
    pass

def calculate_success_rate(data: Data, config: Configuration, model: Model):
    pass


def calculate_statistical_parity(data: Data, config: Configuration, model: Model):
    # Evaluation metrics based on the True Positive Rate of the model for each group within the sensitive features
    pass


def confusion_matrices():
    pass


def calculate_equal_opportunity_difference():
    pass


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


############################################################## Post Processing Bias Mitigation


def post_process_bias_mitigation_eq_odds(
    data_test: Data, model: Model, config: Configuration
):
    pass
