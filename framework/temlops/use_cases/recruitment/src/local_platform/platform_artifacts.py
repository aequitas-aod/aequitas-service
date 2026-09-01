import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from temlops.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
REPORTS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")
DOCUMENTATION_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")

class DataTabular(Data):
    def __init__(self, config):
        super().__init__(config)
    
    def load_dataset(self):
        if self.filetype == "csv":
            dataset = pd.read_csv(os.path.join(DATA_ARTIFACTS_PATH, self.filepath))
        elif self.filetype == "json":
            dataset = pd.read_json(os.path.join(DATA_ARTIFACTS_PATH, self.filepath))
        else:
            dataset = pd.read_parquet(os.path.join(DATA_ARTIFACTS_PATH, self.filepath))
        return dataset
    
    def log_dataset(self, dataset):
        if "resulting_filepath" in self.__dict__:   
            self.filepath = self.resulting_filepath
        if self.filetype == "csv":
            dataset.to_csv(os.path.join(DATA_ARTIFACTS_PATH, self.filepath), index=False)
        elif self.filetype == "json":
            dataset.to_json(os.path.join(DATA_ARTIFACTS_PATH, self.filepath))
        else:
            dataset.to_parquet(os.path.join(DATA_ARTIFACTS_PATH, self.filepath))

class ReportTabular(Report):
    def __init__(self, config):
        super().__init__(config)
        
    def load_report(self):
        if os.path.isfile(os.path.join(REPORTS_ARTIFACTS_PATH, self.filepath)):
            out_path = os.path.join(REPORTS_ARTIFACTS_PATH, self.filepath)
            if self.filetype == "csv":
                self.report = pd.read_csv(out_path)
            elif self.filetype == "json":
                self.report = pd.read_json(out_path)
        else:
            self.report = pd.DataFrame()
        return self.report

    def save_report_stats(self, dataframe):
        out_path = os.path.join(REPORTS_ARTIFACTS_PATH, self.filepath)
        if out_path.endswith(".html"):
            dataframe.save_html(out_path)
        else:
            dataframe.save_json(out_path)        
            
    def save_report_dataframe(self, dataframe):
        out_path = os.path.join(REPORTS_ARTIFACTS_PATH, self.filepath)
        dataframe = pd.concat([dataframe, self.report], ignore_index=True)
        if self.filetype == "csv":
            dataframe.to_csv(out_path, index=False)
        elif self.filetype == "json":
            dataframe.to_json(out_path)
            
    def save_report_image(self, dataframe):
        out_path = os.path.join(REPORTS_ARTIFACTS_PATH, self.filepath)
        if self.filetype == "csv":
            dataframe.to_csv(out_path, index=False)
        elif self.filetype == "json":
            dataframe.to_json(out_path)
            
    def _save_barplot(self, values, algorithms, title, ylabel, filename, ylim=None):
        sns.set_palette("husl")
        plt.style.use("seaborn-v0_8")
        colors = sns.color_palette()[:len(algorithms)]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(algorithms, values, color=colors, alpha=0.9)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)

        for bar in bars:
            height = bar.get_height()
            offset = 0.005 if not ylim else (ylim[1]-ylim[0])*0.01
            ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                    f"{height:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

class ModelTabular(Model):
    def __init__(self, config):
        super().__init__(config)
    
    def load_model(self):
        model_path = os.path.join(MODEL_ARTIFACTS_PATH, self.filepath)
        if os.path.isfile(model_path):
            with open(model_path, "rb") as model_file:
                return pickle.load(model_file)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def save_model(self, model):
        model_path = os.path.join(MODEL_ARTIFACTS_PATH, self.filepath)
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file, pickle.HIGHEST_PROTOCOL)
        with mlflow.start_run(run_name="clf_with_equalized_odds") as run:
            mlflow.sklearn.log_model(model, artifact_path="classifier")


class DocumentationTabular(Documentation):
    def __init__(self, config):
        super().__init__(config)

    def load_content(self):
        doc_path = os.path.join(DOCUMENTATION_ARTIFACTS_PATH, self.filepath)
        if os.path.isfile(doc_path):
            with open(doc_path, "r") as doc_file:
                self.content = doc_file.read()
        else:
            self.content = ""
        return self.content

    def save_content(self, content=None):
        content = content if content is not None else getattr(self, "content", "")
        doc_path = os.path.join(DOCUMENTATION_ARTIFACTS_PATH, self.filepath)
        os.makedirs(os.path.dirname(doc_path), exist_ok=True)
        with open(doc_path, "w") as doc_file:
            doc_file.write(content)