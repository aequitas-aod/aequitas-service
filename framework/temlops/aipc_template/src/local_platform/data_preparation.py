import gdown
import pickle
import numpy as np
import pandas as pd
from datasets import load_dataset
from temlops.src.artifact_types import Data, Configuration, Report, Status
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

""" 
Data preparation stage containing 4 operations:
Data Profiling
Data Validation
Data Preprocessing
Data Documentation
"""
################################################## Data Preprocessing


def tokenize_function(data: Data, config: Configuration):
    tokenizer_model = config.tokenizer_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    return tokenizer(
        data["text"], padding="max_length", truncation=True, max_length=128
    )


def load_data(data: Data, config: Configuration) -> Data:
    raw_dataset = load_dataset("imdb")
    dataset = tokenize_function(raw_dataset, config)

    output_data = Data()
    return dataset


################################################## Data Validation and Quality assurance


def check_common_issues(data):
    common_issues = {
        "special_chars": data["text"].str.contains(r"[^a-zA-Z0-9\s]", regex=True),
        "numbers": data["text"].str.contains(r"\d"),
        "all_caps": data["text"].str.isupper(),
    }
    for issue, mask in common_issues.items():
        data[issue] = mask.sum()
    status = Status
    return None  # TODO Status


def data_quality_report():
    """The data quality report looks at the descriptive statistics and helps visualize relationships in the data. Unlike the data drift report,
    the data quality report can also work for a single dataset.

    You can use it however you like. For example, you can generate and log the data quality snapshot for each model run and save it for future evaluation.
    You can also build a conditional workflow around it: maybe generate an alert or a visual report,
    for example, if you get a high number of new categorical values for a given feature."""
    pass


def validate_cleaned_data():
    pass


def data_drift_detection(current_data, ref_data):
    import datetime
    import json
    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset

    from utils import main_logger

    if isinstance(current_data, list):
        current_data = pd.DataFrame(current_data)
    if ref_data is not None and isinstance(ref_data, list):
        ref_data = pd.DataFrame(ref_data)

    try:
        columns_to_drop = ["dt", "timestamp_in_ms"]
        current_data_clean = current_data.drop(
            columns=[col for col in columns_to_drop if col in current_data.columns],
            errors="ignore",
        )
        ref_data_clean = ref_data.drop(
            columns=[col for col in columns_to_drop if col in ref_data.columns],
            errors="ignore",
        )
        drift_report = Report(metrics=[DataDriftPreset()])
        snapshot = drift_report.run(
            reference_data=ref_data_clean,
            current_data=current_data_clean,
            timestamp=datetime.datetime.now(),
            name="data drift test",
        )
        report_json = snapshot.json()
        report_dict = json.loads(report_json)

        for item in report_dict["metrics"]:
            if isinstance(item.get("value"), (float, int)):
                p_val = item["value"]
                metric_id = item.get("metric_id", "UnknownMetric")
                log_message = f'... ID: {item["id"]} Metric: {metric_id}'
                if p_val < 0.05:
                    main_logger.info(
                        f"{log_message} P-Value: {p_val:.5f} 🚨 drift risk high"
                    )
                else:
                    main_logger.info(
                        f"{log_message} P-Value: {p_val:.5f} ✅ no significant drift"
                    )

    except Exception as e:
        main_logger.error(f"error during detection test: {e}")
