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



def train_model_baseline(data: Data, config: Configuration):
    """ """
    pass

def train_model_reweighing():
    pass

def train_model_fauci():
    pass

############################################################## Model Evaluations - Performance and Fairness metrics


def model_evaluation_accuracy(
    data: Data, config: Configuration, model: Model, report: Report
):
    model = model.load_model()
    # prepare a validation dataset for prediction and predict
    data_test = data.load_dataset()
    pass

def evaluate_fairness_spd(data:Data, model: Model, config: Configuration):    
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

def evaluate_fairness_di(data: Data, model: Model, config: Configuration):
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

