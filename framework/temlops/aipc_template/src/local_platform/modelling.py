import torch
import mlflow
from pickle import dump
from temlops.src.artifact_types import Data, Configuration, Report, Model
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


################################################################ Training model / Fine-tuning


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def fine_tune(tokenized_dataset, config):
    full_train_ds = tokenized_dataset["train"]
    full_eval_ds = tokenized_dataset["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2
    )
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy=config.eval_strategy,
        num_train_epochs=config.num_train_epochs,
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
    fine_tune(data.get_data(), config)


def optimised_training(dataset: Data, config: Configuration):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # rank of matrices (for LoRA)
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # which layers to inject LoRA into
        lora_alpha=64 * 2,  # scaling factor, usually 2x rank
        lora_dropout=0,  # no dropout, increase for regularizaiton
        bias="none",  # bias stays frozen, only learn the low-rank matrices
        use_gradient_checkpointing="unsloth",  # activate custom checkpointing scheme of Unsloth -> higher compute but less GPU memory when backpropagating
    )
    trainer = SFTTrainer(  # supervised fine-tuning trainer
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
        args=SFTConfig(
            per_device_train_batch_size=2,  # each GPU reads 2 tokenized sequences at once
            gradient_accumulation_steps=4,  # accumulate loss for 4 iterations before optimizer step -> effective batch 2 * 4 = 8
            warmup_steps=10,  # linearly "climb" to the learning rate from 0 in the first 10 steps
            max_steps=60,  # max steps before stopping (unless epochs out before that)
            logging_steps=1,  # log every single step
            output_dir="outputs",  # where to store checkpoints, logs etc.
            optim="adamw_8bit",  # 8-bit AdamW optimizer
            num_train_epochs=3,  # number of epochs, unless we reach 60 steps first
        ),
    )
    trainer.train()
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method="lora")


def optimised_fine_tuning_quantisation(data: Data, config: Configuration):
    dataset = data.get_data()
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    training_arguments = TrainingArguments(
        output_dir=model,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="wandb",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,  # ["train"],
        # eval_dataset=dataset["test"],
        peft_config=peft_config,
        max_seq_length=512,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    trainer.train()


############################################################## Evaluations - Performance metrics - Accuracy


def calculate_average_perplexity(
    model: Model, data: Data, config: Configuration
) -> Report:
    model_to_eval = GPT4LMHeadModel.from_pretrained(model.name)
    tokenizer = GPT4Tokenizer.from_pretrained(model.name)

    def calculate_perplexity(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model_to_eval(**inputs)
        return torch.exp(outputs.loss).item()

    sample = data.get_dataset()
    sample_perplexities = sample["text"].apply(calculate_perplexity)
    result = sample_perplexities.mean()
    return result


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
