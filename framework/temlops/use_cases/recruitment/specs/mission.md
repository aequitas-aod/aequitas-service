# Mission

## Product Name
Hiring Classifier — Fairness-Aware Tabular AI Product

## Purpose
Provide a production-grade, fairness-aware classification system for personnel assessment decisions. The product predicts whether a job applicant should be hired based on tabular profile data, while actively detecting and mitigating algorithmic bias against protected demographic groups.

## Problem Statement
Automated hiring tools risk encoding and amplifying historical biases present in training data. This product addresses that risk by embedding fairness constraints at every stage of the ML lifecycle — data preparation, model training, and post-inference adjustment — making bias a first-class engineering concern rather than an afterthought.

## Goals

- Deliver accurate hire/no-hire classification with demographic fairness guarantees.
- Implement multi-stage bias mitigation: pre-processing, in-processing, and post-processing.
- Provide full traceability from raw data through deployed model to monitoring.
- Support both local experimentation and enterprise orchestration on Digital Hub.
- Produce audit-ready documentation (Model Card, Datasheet) for regulatory and ethical review.

## Target Users

| Role | Use |
|---|---|
| ML Engineers | Build, train, and evaluate fairness-aware models |
| MLOps Engineers | Deploy and monitor models in production (KServe, MLflow) |
| Data Scientists | Explore bias/accuracy tradeoffs in experiments |
| AI Product Owners | Review fairness metrics and validate model acceptability |
| Compliance / Ethics Teams | Audit model cards, datasheets, and fairness reports |

## Domain
Personnel Assessment (PA) — hiring and recruitment

## AI Task
Binary classification: hire (1) / no-hire (0)

## Data Modality
Tabular — structured numerical and categorical features (gender, nationality, education, skills, grades)

## Fairness Commitments
The product evaluates and enforces the following fairness criteria across protected groups:

- **Statistical Parity** — difference in selection rates ≤ ±0.10
- **Disparate Impact** — ratio of selection rates within [0.80, 1.20]
- **Equal Opportunity Difference** — difference in true positive rates ≤ ±0.10
- **Average Odds Difference** — average of TPR and FPR differences ≤ ±0.10

## Scope
In scope:
- Tabular classification with structured input features
- Bias mitigation via Reweighing, Grid Search Reduction, and Equalized Odds
- Deployment as a REST API on Kubernetes via KServe
- Production monitoring for data drift and fairness degradation

Out of scope:
- Unstructured data (text, images)
- Generative or ranking models
- Real-time re-training pipelines
