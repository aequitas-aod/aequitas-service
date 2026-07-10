
import os
import yaml
import pickle
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from temlops.src.artifact_types import (
    Data,
    Model,
    Configuration,
    Report,
    Status,
    Documentation,
)
from use_cases.recruitment.src.local_platform.platform_artifacts import (
    DataTabular,
    ReportTabular,
    ModelTabular,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlib import DataFrame
from fairlib.preprocessing import Reweighing, DisparateImpactRemover, LFR
from fairlib.inprocessing import Fauci, AdversarialDebiasing

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




#############################################3 helper functions 

 
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

def _save_barplot(values, algorithms, title, ylabel, filename, ylim=None):
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

