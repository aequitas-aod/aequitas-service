import ollama
import requests
from temlops.use_cases.tabular.src.local_platform.utils import *
from temlops.src.artifact_types import Data, Configuration, Report
from sklearn.metrics import accuracy_score, confusion_matrix

from evidently.report import Report
from evidently.metrics import *
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from unsloth import FastLanguageModel

################################################################################ Model deployment


def model_deployment(config: Configuration):
    model_name = config.name
    model_path = config.path
    ollama.create(model=model_name, from_=model_path)


################################################################################ Model monitoring


def inference_service_unsloth(input_messages):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="finetuned_model", max_seq_length=2048, dtype=None, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    # input_messages = [
    #    {
    #        "role": "user",
    #        "content": "Mike is 30 years old, loves hiking and works as a coder."
    #    },
    # ]
    # Turn messages to tensor and send to GPU
    inputs = tokenizer.apply_chat_template(
        input_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    # Generate model response with max 512 tokens and 0.7 temperature, smallest set of tokens with cumulative probability of >= 0.9 are kept for random sampling
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    response = tokenizer.batch_decode(outputs)[0]
    return response


def inference_service_quantised(input_messages):
    # messages = [
    #    {
    #        "role": "user",
    #        "content": "Mike is 30 years old, loves hiking and works as a coder."
    #    }
    # ]
    inputs = tokenizer.apply_chat_template(
        input_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs,
        max_length=150,
        num_return_sequences=1,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("assistant")[1]


def model_monitoring():
    """mlflow"""
    pass


################################################################################ Production data monitoring


def production_data_monitoring_data_drift(
    prod_data: Data, reference_data: Data
) -> Report:
    report = Report(
        metrics=[
            # Drift in numerical embedding space
            DataDriftPreset(),
            # Drift in predictions
            TargetDriftPreset(),
            # Extra: classification quality (if true labels in production exist)
            ClassificationQualityMetric() if "label" in current_data else None,
        ]
    )
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("text_and_prediction_drift_report.html")


################################################################################ proxy for inference service


def pre_inference_transformation():
    """Adjustments made to production data prior to feeding it into the model"""
    pass


def post_inference_transformation():
    """Adjustments made to the model’s output to prevent data privacy breaches or to filter information."""
    pass
