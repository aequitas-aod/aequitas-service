from library.src.artifact_types import Data, Artifact, Model, Configuration, Report

from fairlearn.metrics import (
    MetricFrame,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    count,
    false_negative_rate_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from fairlearn.reductions import (
    ExponentiatedGradient,
    EqualizedOdds,
    TruePositiveRateParity,
)

################################# fairlearn pre-processing example #################################


################################# fairlearn in-processing example #################################


################################# fairlearn post-processing example #################################


def post_inference_transformations_threshold_EO(
    data_inference: Data, data_postinference: Data, estimator: Model
):
    postprocess_est = ThresholdOptimizer(
        estimator=estimator,
        constraints="equalized_odds",  # Optimize FPR and FNR simultaneously
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )

