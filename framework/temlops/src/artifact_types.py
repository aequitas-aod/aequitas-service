import os
import yaml
import json
import pandas as pd
from pickle import load
from abc import ABC, abstractmethod


class Data():
    """
    Base class for data artifacts. Subclasses should implement specific behavior.
    """
    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)
        self.filepath = config.get("filepath")
        self.filetype = self.filepath.split(".")[-1] if self.filepath and "." in self.filepath else None
        

    def load_dataset(self):
        """Load dataset. Must be implemented by subclasses."""
        pass

    def log_dataset(self, dataset):
        """Log dataset. Must be implemented by subclasses."""
        pass
        

class Report:
    """
    Structured information about the results obtained after applying a function
    """

    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)
        self.filepath = config.get("filepath")
        self.filetype = self.filepath.split(".")[-1] if self.filepath and "." in self.filepath else None
        

    def load_report(self):
        """Must be implemented by subclasses."""
        pass

    def save_report(self, report):
        """Save report. Must be implemented by subclasses."""
        pass


class Model:
    """
    Either a single file or multiple files constituting the model
    """

    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)

    def load_model(self):
        """Must be implemented by subclasses."""
        pass
        
    def save_model(self, model):
        """Must be implemented by subclasses."""
        pass


class Configuration:
    """
    Declarative specifications used to orchestrate the execution of individual components or pipelines
    """

    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)

    def load_config(self):
        with open(self.filepath, "r") as file:
            config = yaml.safe_load(file)
        return config

    def save_config(self, configs):
        with open(self.filepath, "w") as file:
            yaml.dump(configs, file, default_flow_style=False)


class Status:
    """
    A blocking artifact that defines the pipeline execution
    """

    def __init__(self, status_key, status_file):
        self.status_key = status_key
        self.status_file = status_file

    def change_status(self, new_status):
        value = open(self.status_file, "r")
        data = eval(value)

        data[self.status_key] = new_status
        with open(self.status_file, "w") as file:
            file.write("status = ")
            json.dump(data, file, indent=4)


class Documentation:
    """
    Files that contain human-readable content
    """

    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)

    def get_content(self):
        return self.content

    def save_content(self):
        with open(self.filepath, "w") as file:
            file.write("\n\n ".join(self.type, self.content))


class Function:
    """
    An implementation function
    """

    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)


class Service:
    """
    Service encapsulating the model and parameters for serving it, matching evaluation parameters
    """

    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)

class Logs:
    """
    Generated during each phase of the AI lifecycle. Logging of events or Observability services
    """

    def __init__(self, path, level="INFO"):
        self.path = path
        self.level = level
