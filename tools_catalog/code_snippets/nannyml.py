from typing import Any, Iterable, Optional

import nannyml as nml
import pandas as pd

from library.src.artifact_types import Configuration, Data, Report


def _cfg(config: Configuration, key: str, default: Any) -> Any:
	return getattr(config, key, default)


def concept_drift_cbpe(
	reference_data: Data,
	analysis_data: Data,
	config: Configuration,
	estimated_performance_report: Optional[Report] = None,
) -> tuple[Any, Any]:
	"""Estimate concept drift proxy using CBPE.

	Expected config attributes:
	- chunk_size (int, optional)
	- metrics (Iterable[str], optional)
	- y_pred_proba (str, optional)
	- y_pred (str, optional)
	- y_true (str, optional)

	Returns a tuple of (estimator, estimated_performance).
	"""
	reference_df = reference_data.load_dataset()
	analysis_df = analysis_data.load_dataset()

	chunk_size = _cfg(config, "chunk_size", 5000)
	metrics = _cfg(config, "metrics", ("roc_auc",))
	y_pred_proba = _cfg(config, "y_pred_proba", "predicted_probability")
	y_pred = _cfg(config, "y_pred", "prediction")
	y_true = _cfg(config, "y_true", "employed")

	estimator = nml.CBPE(
		problem_type="classification_binary",
		y_pred_proba=y_pred_proba,
		y_pred=y_pred,
		y_true=y_true,
		metrics=list(metrics),
		chunk_size=chunk_size,
	)
	estimator = estimator.fit(reference_df)
	estimated_performance = estimator.estimate(analysis_df)

	if estimated_performance_report is not None:
		estimated_performance_report.save_report(estimated_performance.to_df())

	return estimator, estimated_performance


def data_drift_univariate(
	reference_data: Data,
	analysis_data: Data,
	config: Configuration,
	ranked_features_report: Optional[Report] = None,
) -> tuple[Any, pd.DataFrame]:
	"""Calculate univariate data drift

	Expected config attributes:
	- feature_column_names (list[str], required)
	- chunk_size (int, optional)

	Returns a tuple of (univariate_drift, ranked_features).
	"""
	reference_df = reference_data.load_dataset()
	analysis_df = analysis_data.load_dataset()

	feature_column_names = _cfg(config, "feature_column_names", None)
	if not feature_column_names:
		raise ValueError("config.feature_column_names is required for univariate drift")
	chunk_size = _cfg(config, "chunk_size", 5000)

	univariate_calculator = nml.UnivariateDriftCalculator(
		column_names=feature_column_names,
		chunk_size=chunk_size,
	)
	univariate_calculator.fit(reference_df)
	univariate_drift = univariate_calculator.calculate(analysis_df)

	alert_count_ranker = nml.AlertCountRanker()
	ranked_features = alert_count_ranker.rank(univariate_drift)

	if ranked_features_report is not None:
		ranked_features_report.save_report(ranked_features)

	return univariate_drift, ranked_features


def compare_estimated_vs_realized_performance(
	reference_data: Data,
	analysis_data: Data,
	analysis_targets_data: Data,
	config: Configuration,
	comparison_report: Optional[Report] = None,
) -> tuple[Any, pd.DataFrame]:
	"""Compare estimated (CBPE) and realized performance.

	Expected config attributes:
	- chunk_size (int, optional)
	- metrics (Iterable[str], optional)
	- y_pred_proba (str, optional)
	- y_pred (str, optional)
	- y_true (str, optional)

	Returns a tuple of (comparison_result, analysis_with_targets_df).
	"""
	reference_df = reference_data.load_dataset()
	analysis_df = analysis_data.load_dataset()
	analysis_targets_df = analysis_targets_data.load_dataset()

	chunk_size = _cfg(config, "chunk_size", 5000)
	metrics = _cfg(config, "metrics", ("roc_auc",))
	y_pred_proba = _cfg(config, "y_pred_proba", "predicted_probability")
	y_pred = _cfg(config, "y_pred", "prediction")
	y_true = _cfg(config, "y_true", "employed")

	analysis_with_targets_df = pd.concat([analysis_df, analysis_targets_df], axis=1)

	estimator = nml.CBPE(
		problem_type="classification_binary",
		y_pred_proba=y_pred_proba,
		y_pred=y_pred,
		y_true=y_true,
		metrics=list(metrics),
		chunk_size=chunk_size,
	)
	estimator = estimator.fit(reference_df)
	estimated_performance = estimator.estimate(analysis_df)

	performance_calculator = nml.PerformanceCalculator(
		problem_type="classification_binary",
		y_pred_proba=y_pred_proba,
		y_pred=y_pred,
		y_true=y_true,
		metrics=list(metrics),
		chunk_size=chunk_size,
	)
	performance_calculator.fit(reference_df)
	calculated_performance = performance_calculator.calculate(analysis_with_targets_df)

	comparison_result = estimated_performance.filter(period="analysis").compare(
		calculated_performance
	)

	if comparison_report is not None:
		comparison_report.save_report(comparison_result.to_df())

	return comparison_result, analysis_with_targets_df


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


if __name__ == "__main__":
	import os

	# Example 1: Concept Drift Detection (CBPE)
	print("Example 1: Concept Drift Detection (CBPE)")
	print("-" * 60)

	reference_data = Data(
		filepath="./data/reference.csv",
		platform="local",
		product_name="employment_classifier",
	)
	analysis_data = Data(
		filepath="./data/analysis.csv",
		platform="local",
		product_name="employment_classifier",
	)
	cbpe_config = Configuration(
		config={
			"chunk_size": 5000,
			"metrics": ["roc_auc"],
			"y_pred_proba": "predicted_probability",
			"y_pred": "prediction",
			"y_true": "employed",
		}
	)
	estimated_perf_report = Report(
		filepath="./artifacts/report/estimated_performance.json",
		product_name="employment_classifier",
	)

	estimator, estimated_performance = concept_drift_cbpe(
		reference_data=reference_data,
		analysis_data=analysis_data,
		config=cbpe_config,
		estimated_performance_report=estimated_perf_report,
	)
	print(f"✓ CBPE estimator fitted and performance estimated.")
	print(f"  Result: {type(estimated_performance)}")
	print()

	# Example 2: Data Drift Detection (Univariate)
	print("Example 2: Data Drift Detection (Univariate)")
	print("-" * 60)

	drift_config = Configuration(
		config={
			"feature_column_names": [
				"AGEP",
				"SCHL",
				"MAR",
				"RELP",
				"DIS",
				"ESP",
				"CIT",
				"MIG",
				"MIL",
				"ANC",
				"NATIVITY",
				"DEAR",
				"DEYE",
				"DREM",
				"SEX",
				"RAC1P",
			],
			"chunk_size": 5000,
		}
	)
	ranked_features_report = Report(
		filepath="./artifacts/report/ranked_drift_features.csv",
		product_name="employment_classifier",
	)

	univariate_drift, ranked_features = data_drift_univariate(
		reference_data=reference_data,
		analysis_data=analysis_data,
		config=drift_config,
		ranked_features_report=ranked_features_report,
	)
	print(f"✓ Univariate drift detected on {len(ranked_features)} features.")
	print(f"  Top drifted feature: {ranked_features.head(1) if not ranked_features.empty else 'N/A'}")
	print()

	# Example 3: Compare Estimated vs Realized Performance
	print("Example 3: Compare Estimated vs Realized Performance")
	print("-" * 60)

	analysis_targets_data = Data(
		filepath="./data/analysis_targets.csv",
		platform="local",
		product_name="employment_classifier",
	)
	comparison_config = Configuration(
		config={
			"chunk_size": 5000,
			"metrics": ["roc_auc"],
			"y_pred_proba": "predicted_probability",
			"y_pred": "prediction",
			"y_true": "employed",
		}
	)
	comparison_report = Report(
		filepath="./artifacts/report/estimated_vs_realized.json",
		product_name="employment_classifier",
	)

	comparison_result, analysis_with_targets = compare_estimated_vs_realized_performance(
		reference_data=reference_data,
		analysis_data=analysis_data,
		analysis_targets_data=analysis_targets_data,
		config=comparison_config,
		comparison_report=comparison_report,
	)
	print(f"✓ Comparison computed between estimated and realized performance.")
	print(f"  Result shape: {comparison_result.to_df().shape if hasattr(comparison_result, 'to_df') else 'N/A'}")
	print()

	print("All examples completed successfully!")
