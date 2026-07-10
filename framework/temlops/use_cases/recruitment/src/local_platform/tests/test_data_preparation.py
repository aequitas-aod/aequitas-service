import os
import yaml
import unittest
from library.src.artifact_types import Data, Model, Configuration, Report, Status, Documentation
from library.use_cases.tabular.src.local_platform.platform_artifacts import DataTabular
from library.use_cases.tabular.src.local_platform.data_preparation import load_data, data_profiling_custom, data_profiling_evidently

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
REPORTS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")


class TestAIProduct(unittest.TestCase):
    ai_product_to_test = ["Hiring Classification"]

    def setUp(self):        
        super().setUp()     
        with open(
            "../../../metadata/aipc_local.yaml",
            "r",
        ) as f:
            self.aipc_config = yaml.safe_load(f)   
    
    def test_load_data(self):
        data_artifact = list(
            filter(
                lambda x: x["name"] == "hiring_data_original",
                self.aipc_config["artifacts"]["data"],
            )
        )[0]["spec"]
        data = Data(data_artifact)
        load_data(data)
        
    def test_data_profiling_evidently(self):
        data_artifact = list(
            filter(
                lambda x: x["name"] == "hiring_data_processed",
                self.aipc_config["artifacts"]["data"],
            )
        )[0]["spec"]
        report_artifact = list(
            filter(
                lambda x: x["name"] == "data_profiling_report",
                self.aipc_config["artifacts"]["report"],
            )
        )[0]["spec"]
        data = Data(data_artifact)
        report = Report(report_artifact)

        data_profiling_evidently(data, report)

    def test_data_profiling_custom(self):
        profiling_actions = list(
            filter(
                lambda x: x["name"] == "profiling_actions",
                self.aipc_config["artifacts"]["configuration"],
            )
        )
        data_artifact = list(
            filter(
                lambda x: x["name"] == "hiring_data_processed",
                self.aipc_config["artifacts"]["data"],
            )
        )[0]["spec"]
        data = Data(data_artifact)
        profiling_config = Configuration(config=profiling_actions[0]["config"])

        data_profiling_custom(data, profiling_config)

