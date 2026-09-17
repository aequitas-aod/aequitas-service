
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
    DocumentationTabular
)

import torch
import torch.nn as nn

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




#################################### in-processing techniques
class FauciMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, output_dim):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model_fauci(data: Data, config: Configuration, model: Model) -> Model:
    dataset = DataTabular(data.__dict__).load_dataset()
    X_train = dataset.drop(columns=[config.target_column]).copy(deep=True)
    y_train = dataset[config.target_column].copy(deep=True)

    train_fauci = X_train.copy()
    train_fauci[config.target_column] = y_train
    ds_fauci = DataFrame(train_fauci)
    for col in ds_fauci.columns:
        if ds_fauci[col].dtype == "object" or ds_fauci[col].dtype.name == "category":
            ds_fauci[col], _ = pd.factorize(ds_fauci[col])
    ds_fauci.targets, ds_fauci.sensitive = config.target_column, config.sensitive

    torch.manual_seed(config.random_state)
    base_model = FauciMLP(
        input_dim=X_train.shape[1],
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        output_dim=1,
    )
    fauci_clf = Fauci(
        torchModel=base_model,
        fairness_regularization=config.fairness_regularization,
        regularization_weight=config.regularization_weight,
    )
    fauci_clf.fit(ds_fauci, epochs=config.epochs, batch_size=config.batch_size)

    return ModelTabular(model.__dict__).save_model(fauci_clf)


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

