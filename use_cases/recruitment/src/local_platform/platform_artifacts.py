import os
import pickle
import pandas as pd
from temlops.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
REPORTS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")

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
                return pd.read_csv(out_path)
            elif self.filetype == "json":
                return pd.read_json(out_path)
        return None

    def save_report_stats(self, dataframe):
        out_path = os.path.join(REPORTS_ARTIFACTS_PATH, self.filepath)
        if out_path.endswith(".html"):
            dataframe.save_html(out_path)
        else:
            dataframe.save_json(out_path)        
            
    def save_report_dataframe(self, dataframe):
        out_path = os.path.join(REPORTS_ARTIFACTS_PATH, self.filepath)
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