# Roadmap

## Status Legend
- `done` — implemented and tested
- `in progress` — partially implemented or under active work
- `planned` — scoped but not yet started
- `backlog` — identified but not yet prioritized

---

## Phase 1 — Foundation (done)

Core pipeline stages are implemented for local execution and follow the AIPC YAML specification.

| # | Operation | Status | Notes |
|---|---|---|---|
| 1 | Data Profiling | `done` | YDataProfiler HTML/JSON reports; custom profiling actions |
| 2 | Data Validation | `done` | Schema checks, missing value detection, constraint validation |
| 3 | Data Preprocessing | `done` | One-hot encoding, train/valid/test split (70-20-10) |
| 4 | Data Documentation | `done` | Datasheet (Gebru et al.) in `docs/Datasheet.md` |
| 5 | Feature Engineering | `done` | Permutation-based feature importance; encoded feature set |
| 6 | Model Training | `done` | RidgeClassifier baseline; reweighing and Grid Search Reduction variants |
| 7 | Model Evaluation | `done` | Accuracy + 7 fairness metrics across demographic groups |
| 8 | Model Validation | `done` | Threshold-based pass/fail for fairness and accuracy |
| 9 | Model Documentation | `done` | Model Card in `docs/ModelCard.md` |
| 10 | Model Deployment | `done` | KServe InferenceService; `configs/kserve.yaml` |
| 11 | Model Monitoring | `done` | MLflow tracking; FastAPI prediction logging endpoint |
| 12 | Production Data Monitoring | `done` | Evidently drift detection integrated into operationalization |
| 13 | System Monitoring | `in progress` | Basic logging in place; latency and error-rate alerting not yet wired |
| 14 | Pre-inference Transformations | `done` | Input encoding applied before inference |
| 15 | Post-inference Transformations | `done` | Equalized Odds adjustment applied to predictions |

---

## Phase 2 — Platform Integration (in progress)

Lift the local pipeline to the Digital Hub (enterprise Kubernetes + Argo Workflows).

| Item | Status | Notes |
|---|---|---|
| Digital Hub runtime wrappers (`@handler`) | `done` | `src/dh_platform/` modules complete |
| Argo Workflow DAG definition | `done` | `pipeline.py` with Hera DSL |
| `aipc_dh.yaml` pipeline spec | `done` | DH-specific YAML mirroring local spec |
| End-to-end DH pipeline execution | `in progress` | Needs validation run on target cluster |
| KServe deployment from DH artifacts | `planned` | Automate InferenceService creation post-training |
| MLflow server on DH | `planned` | Connect DH experiment runs to centralized MLflow |

---

## Phase 3 — Robustness & Advanced Fairness (planned)

Extend the product with adversarial robustness and richer fairness techniques.

| Item | Status | Notes |
|---|---|---|
| Robustness layer operations (`robustness_layer_ops.yaml`) | `planned` | YAML spec exists; implementation pending |
| Adversarial input testing | `planned` | Validate model stability under perturbed inputs |
| Intersectional fairness analysis | `planned` | Evaluate fairness across combinations of protected attributes |
| Fairlearn `ThresholdOptimizer` integration | `planned` | Alternative post-processing technique |
| Counterfactual explanations (aix360) | `planned` | "What would need to change for a different outcome?" |

---

## Phase 4 — Governance & Observability (planned)

Strengthen audit trails, automated alerts, and compliance reporting.

| Item | Status | Notes |
|---|---|---|
| Automated fairness regression tests in CI | `planned` | Gate model updates on fairness thresholds |
| System monitoring — latency and error alerts | `planned` | Complete Stage 13; integrate with alerting (Prometheus/Grafana) |
| Fairness dashboard | `planned` | Real-time fairness metric visualization for production |
| Data lineage tracking | `planned` | Trace each artifact back to its source dataset and preprocessing step |
| Automated model card generation | `planned` | Generate `ModelCard.md` from pipeline run metadata |
| Regulatory compliance report | `backlog` | EU AI Act / EEOC-style audit artifact |

---

## Phase 5 — Extensibility (backlog)

Generalize the framework to support additional use cases and modalities.

| Item | Status | Notes |
|---|---|---|
| Multi-class classification support | `backlog` | Extend beyond binary hire/no-hire |
| Regression task support | `backlog` | Adapt fairness metrics for continuous outputs |
| Feature store integration | `backlog` | Decouple feature engineering from training pipeline |
| Online learning / incremental retraining | `backlog` | Update model incrementally as production data arrives |

---

## Known Technical Debt

| Item | Priority | Notes |
|---|---|---|
| `README.md` is empty | High | Add product overview and quickstart |
| No CI/CD pipeline configured | High | Tests are not run automatically on commit |
| `tests/` contains only one test file | Medium | Coverage is minimal; expand to cover all operations |
| Hard-coded data download URL (`gdown`) | Medium | Should be configurable via YAML or env var |
| System monitoring (Stage 13) incomplete | Medium | Latency tracking and alerting not yet implemented |
