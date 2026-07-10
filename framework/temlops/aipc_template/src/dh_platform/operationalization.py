import ollama
import requests
from utils import *
import digitalhub as dh
from artifact_types import Data, Configuration, Report
from sklearn.metrics import accuracy_score, confusion_matrix

project = dh.get_or_create_project("nlp_use_case")

################################################################################ Model deployment


def model_deploy(config: Configuration):
    llm_function = project.new_function(
        "llm_classification",
        kind="huggingfaceserve",
        model_name="mymodel",
        path="huggingface://distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    llm_run = llm_function.run(action="serve", profile="1xa100", wait=True)


################################################################################ Model monitoring
# experiments tracking: efficient


def model_monitoring():
    """mlflow"""
    pass


################################################################################ proxy for inference service


def pre_inference_transformation():
    pass


def post_inference_transformation():
    pass
