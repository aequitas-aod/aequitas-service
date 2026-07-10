import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from library.src.artifact_types import Data, Artifact, Model, Configuration, Report


def data_profiling_detect_data_drift(
    reference_data: Data,
    current_data: Data,
):
    """
    Function to detect data drift
    """
    # Create a report with the DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])

    # Run the report with the provided data
    report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    # Return the results
    return report.show()
