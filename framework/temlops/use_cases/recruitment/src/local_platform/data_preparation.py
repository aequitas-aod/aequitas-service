import sys
sys.path.append("../../../../")

import os
import yaml
import json
import openml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Any, Tuple
from temlops.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation
from temlops.use_cases.recruitment.src.local_platform.platform_artifacts import DataTabular, ReportTabular

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

def load_data(data: Data, data_processed: Data) -> Data:
    adult_ds = openml.datasets.get_dataset(data.filepath)
    adult_df, *_ = adult_ds.get_data(dataset_format="dataframe")

    adult_df.rename(columns={"class": "income"}, inplace=True)
    adult_df.drop(columns=["fnlwgt"], inplace=True)
    
    DataTabular(data_processed.__dict__).log_dataset(adult_df)
    return data_processed


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
    data_train: Data, 
    config: Configuration
    ):
    data = DataTabular(data_train.__dict__)
    df = data.load_dataset()
    results = []
    for act in config.actions.split(","):
        action = act.strip()
        result_meta = {"action": action}
        if action == "summary":
            result_meta["n_rows"] = int(df.shape[0])
            result_meta["n_cols"] = int(df.shape[1])
            result_meta["memory_mb"] = float(
                df.memory_usage(deep=True).sum() / (1024 * 1024)
            )
        elif action == "dtypes":
            result_meta["dtypes"] = df.dtypes.apply(lambda x: str(x)).to_dict()
        elif action == "missing_values":
            mv = df.isna().sum()
            result_meta["missing_count"] = mv.to_dict()
            result_meta["missing_pct"] = (mv / len(df)).round(4).to_dict()
        elif action == "unique_values":
            cols = df.columns.tolist()
            uniques = {}
            for c in cols:
                if c in df.columns:
                    uniques[c] = {
                        "unique_count": int(df[c].nunique()),
                        # "top": df[c].mode().iloc[0] if not df[c].mode().empty else None,
                    }
            result_meta["unique"] = uniques
        elif action == "correlations":
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] >= 2:
                corr = numeric.corr()
                result_meta["correlation_head"] = corr.round(3).iloc[:5, :5].to_dict()
                # Save full correlation to file
                corr.to_csv(
                    os.path.join(
                        REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_correlation.csv"
                    )
                )
                # Save correlation image
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title(f"Correlation matrix {config.dataset_name}")
                img_path = os.path.join(
                    REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_correlation.png"
                )
                fig.savefig(img_path, bbox_inches="tight")
                plt.close(fig)

                result_meta["correlation_csv"] = (
                    f"{config.dataset_name}_correlation.csv"
                )
                result_meta["correlation_png"] = img_path
            else:
                result_meta["note"] = "Not enough numeric columns for correlation."
        elif action == "histograms":
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            bins = 30
            img_paths = []
            for c in cols:
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    fig, ax = plt.subplots()
                    df[c].dropna().hist(bins=bins, ax=ax)
                    ax.set_title(f"Histogram {c}")
                    img_path = os.path.join(
                        REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_hist_{c}.png"
                    )
                    fig.savefig(img_path, bbox_inches="tight")
                    plt.close(fig)
                    img_paths.append(img_path)
            result_meta["histograms"] = img_paths
        elif action == "outliers":
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            outliers = {}
            for c in cols:
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    q1 = df[c].quantile(0.25)
                    q3 = df[c].quantile(0.75)
                    iqr = q3 - q1
                    low = q1 - 1.5 * iqr
                    high = q3 + 1.5 * iqr
                    mask = (df[c] < low) | (df[c] > high)
                    outliers[c] = {
                        "n_outliers": int(mask.sum()),
                        "low": float(low),
                        "high": float(high),
                    }
            result_meta["outliers"] = outliers
        elif action == "profile_report":
            title = f"Profile report {config.dataset_name}"
            profile = ProfileReport(df, title=title, explorative=True)
            out_html = os.path.join(
                REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_profile.html"
            )
            profile.to_file(out_html)
            result_meta["profile_html"] = out_html
        else:
            result_meta["error"] = "action not implemented"
        results.append(result_meta)
        # Save results summary
        report_final_path = os.path.join(
            REPORTS_ARTIFACTS_PATH, f"{config.dataset_name}_final_report.json"
        )
        with open(report_final_path, "w", encoding="utf-8") as file:
            json.dump(
                {"actions": config.actions, "results": results},
                file,
                indent=2,
                ensure_ascii=False,
            )

    return results
 
########################################################### Data Validation
    
def data_validation_check_quantity(data: Data, config: Configuration, output_status: Status) -> Status:
    pass

def data_validation_demographics_qty(data: Data, config: Configuration, output_status: Status) -> Status:
    pass

def data_drift_detection(data: Data, config: Report):
    pass

def data_drift_status(data: Data, output_status: Status):
    pass



########################################################### Data Preprocessing

def split_train_valid_test_data(data: Data, config: Configuration, data_train: Data, data_test: Data, data_valid: Data) -> Tuple[Data, Data, Data]:
    dataset = DataTabular(data.__dict__).load_dataset()
    for col in dataset.columns:
        if dataset[col].dtype == "object" or dataset[col].dtype.name == "category":
            dataset[col], _ = pd.factorize(dataset[col])
    # First split: train+val vs test
    X_train_val, X_test = train_test_split(
        dataset, test_size=config.test_size, random_state=config.random_state
    )
    # Second split: train vs validation
    X_train, X_val = train_test_split(
        X_train_val, test_size=config.valid_size, random_state=config.random_state  # 0.25 x 0.8 = 0.2
    )
    DataTabular(data_train.__dict__).log_dataset(X_train)
    DataTabular(data_test.__dict__).log_dataset(X_test)
    DataTabular(data_valid.__dict__).log_dataset(X_val)
        
    return data_train, data_test, data_valid


def preprocess_train_data(data_input: Data, data_output: Data):
    pass


def preprocess_reweighing(data_input: Data, data_output: Data):
    pass

########################################################### Data Documentation

def data_card_generation(data: Data, documentation: Documentation):
    pass





if __name__ == "__main__":
    def _resolve_vars(specs_list, data_artifacts, config_artifacts, model_artifacts):
        vars = {}
        for item in specs_list:
            artifact_name = list(item.values())[0]
            key = list(item.keys())[0]
            match = next((a for a in data_artifacts if a["name"] == artifact_name), None)
            if match:
                vars[key] = Data(**{k: v for k, v in match.items() if k != "name"})
            match = next((a for a in config_artifacts if a["name"] == artifact_name), None)
            if match:
                vars[key] = Configuration(**{k: v for k, v in match.items() if k != "name"})
            match = next((a for a in model_artifacts if a["name"] == artifact_name), None)
            if match:
                vars[key] = Model(**{k: v for k, v in match.items() if k != "name"})
        return vars

    def run_operation(
        operation,
        data_artifacts,
        model_artifacts,
        config_artifacts
    ):
        specs = operation["implementation"]["spec"]
        method_name = specs["method_name"]

        input_vars = _resolve_vars(specs["inputs"], data_artifacts, config_artifacts, model_artifacts)
        input_vars.update(_resolve_vars(specs["outputs"], data_artifacts, config_artifacts, model_artifacts))
        print(input_vars)

        func = globals()[method_name]
        func(**input_vars)
       
    with open(
        "../../metadata/aipc_local.yaml",
        "r",
    ) as f:
        aipc_config = yaml.safe_load(f)
    operation = list(
        filter(
            lambda x: x["id"] == "data_profiling_custom",
            aipc_config["operations"],
        )
    )[0]
    data_artifacts = aipc_config["artifacts"]["data"]
    model_artifacts = aipc_config["artifacts"]["model"]
    config_artifacts = aipc_config["artifacts"]["configuration"]
    run_operation(
        operation,
        data_artifacts,
        model_artifacts,
        config_artifacts
    )