import sys

sys.path.append("../../../../")

import os
import yaml
import pandas as pd


from temlops.src.artifact_types import (
    Data,
    Model,
    Configuration,
    Report,
    Status,
    Documentation,
)
from temlops.use_cases.recruitment.src.local_platform.platform_artifacts import (
    DataTabular,
    ReportTabular,
    ModelTabular,
    DocumentationTabular,
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlib import DataFrame
from fairlib.preprocessing import Reweighing, DisparateImpactRemover, LFR
from fairlib.inprocessing import Fauci, AdversarialDebiasing

FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")
REPORT_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")
STATUS_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "status")


#################################################### Model Training ####################################################


def train_model_baseline(data: Data, config: Configuration, model: Model) -> Model:
    dataset = DataTabular(data.__dict__).load_dataset()
    X_train = dataset.drop(columns=[config.target_column]).copy(deep=True)
    y_train = dataset[config.target_column].copy(deep=True)

    base_clf = train_classifier(
        X_train, y_train, random_state=config.random_state, max_iter=config.max_iter
    )
    return ModelTabular(model.__dict__).save_model(base_clf)


#################################### pre-processing techniques
def train_model_reweighing(data: Data, config: Configuration, model: Model) -> Model:
    dataset = DataTabular(data.__dict__).load_dataset()
    X_train = dataset.drop(columns=[config.target_column]).copy(deep=True)
    y_train = dataset[config.target_column].copy(deep=True)

    train_rw = X_train.copy()
    train_rw[config.target_column] = y_train
    ds_rw = DataFrame(train_rw)
    ds_rw.targets, ds_rw.sensitive = config.target_column, config.sensitive
    rw_proc = Reweighing()
    ds_rw_t = rw_proc.fit_transform(ds_rw)

    rw_clf = train_classifier(X_train, y_train, sample_weight=ds_rw_t["weights"].values)
    return ModelTabular(model.__dict__).save_model(rw_clf)


def train_model_disparate_impact_remover(
    data: Data, config: Configuration, model: Model
) -> Model:
    # Requires sensitive attribute information during both training and inference
    dataset = DataTabular(data.__dict__).load_dataset()
    X_train = dataset.drop(columns=[config.target_column]).copy(deep=True)
    y_train = dataset[config.target_column].copy(deep=True)

    train_dir = X_train.copy()
    train_dir[config.target_column] = y_train
    ds_dir = DataFrame(train_dir)
    ds_dir.targets, ds_dir.sensitive = config.target_column, config.sensitive
    dir_proc = DisparateImpactRemover(repair_level=1.0)
    train_dir_t = dir_proc.fit_transform(ds_dir).drop(columns=[config.sensitive])
    dir_clf = train_classifier(train_dir_t, y_train)
    return ModelTabular(model.__dict__).save_model(dir_clf)


def train_model_learning_fair_representations(
    data: Data, config: Configuration, model: Model
) -> Model:
    dataset = DataTabular(data.__dict__).load_dataset()
    X_train = dataset.drop(columns=[config.target_column]).copy(deep=True)
    y_train = dataset[config.target_column].copy(deep=True)

    latent_dim = 8
    lfr_proc = LFR(
        input_dim=X_train.shape[1], latent_dim=latent_dim, output_dim=X_train.shape[1]
    )

    # Prepare data for LFR
    train_lfr_df = X_train.copy()
    train_lfr_df[config.target_column] = y_train
    ds_lfr_train = DataFrame(train_lfr_df)
    ds_lfr_train.targets, ds_lfr_train.sensitive = (
        config.target_column,
        config.sensitive,
    )

    ds_lfr_latent = lfr_proc.fit_transform(ds_lfr_train, epochs=60, learning_rate=0.001)
    X_train_lfr = pd.DataFrame(ds_lfr_latent.values, columns=ds_lfr_latent.columns)
    lfr_clf = train_classifier(X_train_lfr, y_train)

    return ModelTabular(model.__dict__).save_model(lfr_clf)


#################################### in-processing techniques
def train_model_fauci():
    pass


def train_model_adversarial_debiasing():
    pass


############################################################## Model Evaluations - Performance and Fairness metrics


def model_evaluation_performance(
    data_test: Data, config: Configuration, model: Model, report: Report
) -> Report:
    model_test = ModelTabular(model.__dict__).load_model()
    dataset_test = DataTabular(data_test.__dict__).load_dataset()
    report = ReportTabular(report.__dict__)
    report.load_report()
    X_test = dataset_test.drop(columns=[config.target_column]).copy(deep=True)
    y_test = dataset_test[config.target_column].copy(deep=True)

    base_pred = model_test.predict(X_test)
    base_acc = accuracy_score(y_test, base_pred)
    report.save_report_dataframe(
        pd.DataFrame(
            [{"algorithm": model.filepath.split(".")[0], "accuracy": base_acc}]
        )
    )
    return report


def model_evaluation_fairness(
    data_test: Data, config: Configuration, model: Model, report: Report
) -> Report:
    model_test = ModelTabular(model.__dict__).load_model()
    report = ReportTabular(report.__dict__)
    data = DataTabular(data_test.__dict__)
    dataset = data.load_dataset()
    X_test = dataset.drop(config.target, axis=1)
    y_pred = model_test.predict(X_test)

    spd, di = evaluate_fairness(
        X_test,
        y_pred,
        config.target,
        config.sensitive,
        config.positive_target,
        config.favored_class,
    )
    report.save_report_dataframe(
        pd.DataFrame(
            [{"algorithm": model.filepath.split(".")[0], "spd": spd, "di": di}]
        )
    )
    return report


def model_evaluation_fairness_disparate_impact_remover(
    data_test: Data, config: Configuration, model: Model, report: Report
) -> Report:
    model_test = ModelTabular(model.__dict__).load_model()
    report = ReportTabular(report.__dict__)
    data = DataTabular(data_test.__dict__)
    dataset = data.load_dataset()
    X_test = dataset.drop(columns=[config.target], axis=1)
    dir_pred = model_test.predict(X_test.drop(columns=[config.sensitive], axis=1))

    dir_spd, dir_di = evaluate_fairness(
        X_test,
        dir_pred,
        config.target,
        config.sensitive,
        config.positive_target,
        config.favored_class,
    )
    report.save_report_dataframe(
        pd.DataFrame(
            [{"algorithm": model.filepath.split(".")[0], "spd": dir_spd, "di": dir_di}]
        )
    )
    return report


def model_evaluation_fairness_lfr(
    data_test: Data, data_train: Data, config: Configuration, model: Model, report: Report
) -> Report:
    model_test = ModelTabular(model.__dict__).load_model()
    report = ReportTabular(report.__dict__)
    data = DataTabular(data_test.__dict__)
    dataset_test = data.load_dataset()
    data = DataTabular(data_train.__dict__)
    dataset_train = data.load_dataset()
    X_test = dataset_test.drop(columns=[config.target], axis=1)
    X_train = dataset_train.drop(columns=[config.target], axis=1)
    y_test = dataset_test[config.target]
    y_train = dataset_train[config.target]

    # Trasform test data
    latent_dim = 8
    lfr_proc = LFR(
        input_dim=X_train.shape[1], latent_dim=latent_dim, output_dim=X_train.shape[1]
    )
    train_lfr_df = X_train.copy(); train_lfr_df[config.target] = y_train
    ds_lfr_train = DataFrame(train_lfr_df); ds_lfr_train.targets, ds_lfr_train.sensitive = config.target, config.sensitive
    ds_lfr_latent = lfr_proc.fit_transform(ds_lfr_train, epochs=60, learning_rate=0.001)
    
    test_lfr_df = X_test.copy()
    test_lfr_df[config.target] = y_test
    ds_lfr_test = DataFrame(test_lfr_df)
    ds_lfr_test.targets, ds_lfr_test.sensitive = config.target, config.sensitive
    X_test_lfr_df = lfr_proc.transform(ds_lfr_test)
    X_test_lfr = pd.DataFrame(X_test_lfr_df.values, columns=X_test_lfr_df.columns)

    lfr_pred = model_test.predict(X_test_lfr)
    lfr_spd, lfr_di = evaluate_fairness(
        X_test,
        lfr_pred,
        config.target,
        config.sensitive,
        config.positive_target,
        config.favored_class,
    )
    report.save_report_dataframe(
        pd.DataFrame(
            [{"algorithm": model.filepath.split(".")[0], "spd": lfr_spd, "di": lfr_di}]
        )
    )
    return report


def check_all_fairness_metrics(report: Report, report2: Report) -> Report:
    report = ReportTabular(report.__dict__)
    report_data = ReportTabular(report.__dict__).load_report()
    algorithms = report_data["algorithm"].tolist()

    #accuracy = report_data["accuracy"].tolist()
    spd_abs = report_data["spd"].abs().tolist()
    di_abs = (report_data["di"] - 1).abs().tolist()
    
    #report._save_barplot(accuracy, algorithms,
    #              title="Accuracy – Pre-processing Algorithms",
    #              ylabel="Accuracy",
    #              filename="preprocessing_accuracy_comparison.png",
    #              ylim=(0.75, 0.85))

    report._save_barplot(spd_abs, algorithms,
                  title="|Statistical Parity Difference| – Lower is Better",
                  ylabel="|SPD|",
                  filename=os.path.join(REPORT_ARTIFACTS_PATH, "preprocessing_spd_comparison.png"))

    report._save_barplot(di_abs, algorithms,
                  title="|Disparate Impact − 1| – Lower is Better",
                  ylabel="|DI−1|",
                  filename=os.path.join(REPORT_ARTIFACTS_PATH, "preprocessing_di_comparison.png"))


############################################# Model Documentation

def model_documentation(report: Report, documentation: Documentation) -> Documentation:
    report_table = ReportTabular(report.__dict__)
    report_table.load_report()

    documentation.content = _render_model_card(report_table.report, documentation)
    DocumentationTabular(documentation.__dict__).save_content(documentation.content)
    return documentation


############# Model Validation


def model_validation_baseline(report: Report, config: Configuration, status: Status):
    pass


############################################################## Post Process Bias Mitigation Techniques


def bias_mitigation_post_process_train(data: Data, config: Configuration, report: Report):
    pass


############################################################### helper functions


def train_classifier(X, y, sample_weight=None, random_state=42, max_iter=1000):
    clf = LogisticRegression(random_state=random_state, max_iter=max_iter)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def evaluate_fairness(
    X_test, y_pred, target, sensitive, positive_target=1, favored_class=0
):
    X_eval = X_test.copy()
    X_eval[target] = y_pred
    ds_eval = DataFrame(X_eval)
    ds_eval.targets, ds_eval.sensitive = target, sensitive

    spd = ds_eval.statistical_parity_difference()[
        {target: positive_target, sensitive: favored_class}
    ]
    di = ds_eval.disparate_impact()[{target: positive_target, sensitive: favored_class}]
    return spd, di

def _model_card_field(documentation, attr, default="More information needed."):
    value = getattr(documentation, attr, None)
    return value if value not in (None, "") else default


def _results_table(results: pd.DataFrame) -> str:
    if results is None or results.empty:
        return "More information needed."

    columns = list(results.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| "
        + " | ".join(
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in row
        )
        + " |"
        for _, row in results.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _render_model_card(results: pd.DataFrame, documentation: Documentation) -> str:
    field = lambda attr, default="More information needed.": _model_card_field(
        documentation, attr, default
    )

    return f"""# {field("model_id", "Recruitment Hiring Classifier")}

# Table of Contents

- [Model Details](#model-details)
- [Uses](#uses)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Model Examination](#model-examination)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications](#technical-specifications)
- [Citation](#citation)
- [More Information](#more-information)
- [Model Card Authors](#model-card-authors)
- [Model Card Contact](#model-card-contact)

# Model Details

## Model Description

{field("model_description")}

- **Developed by:** {field("developers")}
- **Model type:** {field("model_type", "Binary classifier")}
- **License:** {field("license")}

# Uses

## Direct Use

{field("direct_use")}

## Out-of-Scope Use

{field("out_of_scope_use")}

# Bias, Risks, and Limitations

{field("bias_risks_limitations")}

## Recommendations

{field(
        "bias_recommendations",
        "Users (both direct and downstream) should be made aware of the risks, "
        "biases and limitations of the model.",
    )}

# Training Details

## Training Data

{field("training_data")}

## Training Procedure

{field("training_procedure")}

# Evaluation

## Testing Data, Factors & Metrics

### Testing Data

{field("testing_data")}

### Metrics

{field(
        "testing_metrics",
        "Accuracy, Statistical Parity Difference (SPD), Disparate Impact (DI).",
    )}

## Results

{_results_table(results)}

# Model Examination

{field("model_examination")}

# Environmental Impact

- **Hardware Type:** {field("hardware")}
- **Hours used:** {field("hours_used")}
- **Cloud Provider:** {field("cloud_provider")}
- **Compute Region:** {field("cloud_region")}
- **Carbon Emitted:** {field("co2_emitted")}

# Technical Specifications

## Model Architecture and Objective

{field("model_specs")}

## Compute Infrastructure

{field("compute_infrastructure")}

# Citation

{field("citation")}

# More Information

{field("more_information")}

# Model Card Authors

{field("model_card_authors")}

# Model Card Contact

{field("model_card_contact")}
"""



    
if __name__ == "__main__":

    def _resolve_vars(
        specs_list, data_artifacts, config_artifacts, model_artifacts, report_artifacts
    ):
        vars = {}
        for item in specs_list:
            artifact_name = list(item.values())[0]
            key = list(item.keys())[0]
            match = next(
                (a for a in data_artifacts if a["name"] == artifact_name), None
            )
            if match:
                vars[key] = Data(**{k: v for k, v in match.items() if k != "name"})
            match = next(
                (a for a in config_artifacts if a["name"] == artifact_name), None
            )
            if match:
                vars[key] = Configuration(
                    **{k: v for k, v in match.items() if k != "name"}
                )
            match = next(
                (a for a in model_artifacts if a["name"] == artifact_name), None
            )
            if match:
                vars[key] = Model(**{k: v for k, v in match.items() if k != "name"})
            match = next(
                (a for a in report_artifacts if a["name"] == artifact_name), None
            )
            if match:
                vars[key] = Report(**{k: v for k, v in match.items() if k != "name"})
        return vars

    def run_operation(
        operation, data_artifacts, model_artifacts, config_artifacts, report_artifacts
    ):
        specs = operation["implementation"]["spec"]
        method_name = specs["method_name"]

        input_vars = _resolve_vars(
            specs["inputs"],
            data_artifacts,
            config_artifacts,
            model_artifacts,
            report_artifacts,
        )
        input_vars.update(
            _resolve_vars(
                specs["outputs"],
                data_artifacts,
                config_artifacts,
                model_artifacts,
                report_artifacts,
            )
        )
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
            lambda x: x["id"] == "check_all_fairness_metrics",
            aipc_config["operations"],
        )
    )[0]
    data_artifacts = aipc_config["artifacts"]["data"]
    model_artifacts = aipc_config["artifacts"]["model"]
    config_artifacts = aipc_config["artifacts"]["configuration"]
    report_artifacts = aipc_config["artifacts"]["report"]
    run_operation(
        operation, data_artifacts, model_artifacts, config_artifacts, report_artifacts
    )
