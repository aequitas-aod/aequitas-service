import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

from framework.temlops.src.artifact_types import Configuration, Data, Logs, Model, Report

def explainability_fairness_what_if_analysis(
	input_data: Data,
	model: Model,
	config: Configuration,
	report: Report
 ):
	"""
	Build and return a What-If Tool widget from TEMLOPS artifacts.

	Input artifacts:
	- input_data (Data): dataset artifact with feature columns (+ optional target)
	- model (Model): trained model artifact (must support predict or predict_proba)
	- config (Configuration): YAML config artifact with keys:
		- target_column (optional)
		- sample_size (optional, default 200)
		- model_type (optional: "classification" | "regression")
		- random_state (optional, default 42)

	Output artifact:
	- report (Report): JSON report describing the generated WIT session setup
	"""
	dataset = input_data.load_dataset()
	estimator = model.load_model()

	target_column = config.target_column
	sample_size = config.sample_size
	random_state = config.random_state

	if target_column and target_column in dataset.columns:
		features_df = dataset.drop(columns=[target_column])
	else:
		features_df = dataset.copy()

	if sample_size < len(features_df):
		features_df = features_df.sample(n=sample_size, random_state=random_state)

	examples = features_df.to_dict(orient="records")

	if hasattr(estimator, "predict_proba"):
		model_type = config.model_type or "classification"

		def predict_fn(rows):
			frame = pd.DataFrame(rows)
			return estimator.predict_proba(frame).tolist()
	else:
		model_type = config.model_type or "regression"

		def predict_fn(rows):
			frame = pd.DataFrame(rows)
			return estimator.predict(frame).reshape(-1, 1).tolist()

	builder = (
		WitConfigBuilder(examples)
		.set_custom_predict_fn(predict_fn)
		.set_model_type(model_type)
	)

	widget = WitWidget(builder, height=600)
	return widget
