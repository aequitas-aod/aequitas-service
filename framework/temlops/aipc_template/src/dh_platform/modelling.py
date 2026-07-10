import torch
import mlflow
from pickle import dump
from artifact_types import Data, Configuration, Report, Model

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    GPT4LMHeadModel,
    GPT4Tokenizer,
)

################################################################ Training model / Fine-tuning


def prepare_dataset():
    raw_datasets = load_dataset("imdb")
    return raw_datasets


def tokenize_function(example):
    checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=128
    )


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def fine_tune(raw_datasets):
    tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)
    full_train_ds = tokenized_dataset["train"]
    full_eval_ds = tokenized_dataset["test"]

    checkpoint = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    training_args = TrainingArguments(
        output_dir="ft_model", eval_strategy="epoch", num_train_epochs=5
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_train_ds,
        eval_dataset=full_eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def train_model(data: Data, config: Configuration):
    # the operation specified in the aipc.yaml file
    """
    TODO mlflow logging
    """


############################################################## Evaluations - Performance metrics - Accuracy


def calculate_perplexity(model: Model, data: Data, config: Configuration) -> Report:
    model_to_eval = GPT4LMHeadModel.from_pretrained("GPT4")
    tokenizer = GPT4Tokenizer.from_pretrained("GPT4")

    inputs = tokenizer(
        data.get_dataset(), return_tensors="pt", truncation=True, max_length=1024
    )
    with torch.no_grad():
        outputs = model_to_eval(inputs, labels=inputs["input_ids"])
    report = Report()
    return torch.exp(outputs.loss).item()


############################################################## Accuracy and fairness evaluation metrics


def model_evaluation(data: Data, config: Configuration, model: Model):
    model_name = model.name
    model_version = model.vesion

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

    # prepare a validation dataset for prediction and predict
    data = data.get_dataset()
    y_pred_new = model.predict(data)
    # TODO compare metrics and generate report (csv/json)


############################################################## Bias Mitigation techniques


def bias_mitigation_in_process_train(data: Data, config: Configuration, report: Report):
    pass


############################################################## Explainability


def explain_model_predictions(blackbox_model, X_train, y_test):
    pass


############################################################## Robustness


def robustness_evaluation():
    pass
