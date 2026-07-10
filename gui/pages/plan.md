You are an AI coding agent tasked with implementing a structured AI product following a modular MLOps framework.

Your goal is to implement a set of operations that correspond to the stages of the AI lifecycle. Each operation must be implemented as a reusable, modular, and well-documented component, following best practices for maintainability, transparency, and extensibility.

## GENERAL REQUIREMENTS

- Each operation must be implemented as an independent, reusable module (e.g., Python function or class).
- Clearly define:
  - Input artifacts
  - Output artifacts
  - Configuration parameters
- Follow a declarative approach: each operation should be describable via a YAML specification.
- Ensure traceability between:
  - requirements
  - operations
  - artifacts
- Include logging, validation, and documentation for each operation.
- Where applicable, suggest or integrate appropriate open-source libraries.

---

## STAGE 1: DATA PREPARATION

### 1. Data Profiling
- Generate descriptive statistics and data quality reports.
- Output: profiling report (JSON/HTML)

### 2. Data Validation
- Validate schema, missing values, and constraints.
- Output: validation status + issues

### 3. Data Preprocessing
- Perform:
  - data cleaning
  - type conversion
  - normalization
  - bias analysis (if applicable)
- Output: processed dataset

### 4. Data Documentation
- Generate human-readable documentation describing:
  - dataset origin
  - preprocessing steps
- Output: documentation artifact

---

## STAGE 2: MODELLING

### 5. Feature Engineering and Selection
- Apply transformations and feature importance analysis.
- Output:
  - transformed dataset
  - feature report

### 6. Model Training
- Train or fine-tune a model.
- Ensure reproducibility via configuration.
- Output: trained model artifact (function or serialized model)

### 7. Model Evaluation
- Evaluate model using appropriate metrics.
- Include fairness metrics if required.
- Output: evaluation report

### 8. Model Validation
- Validate model against predefined thresholds.
- Output: validation status (pass/fail)

### 9. Model Documentation
- Generate model card including:
  - performance
  - assumptions
  - limitations
- Output: documentation artifact

---

## STAGE 3: OPERATIONALIZATION

### 10. Model Deployment
- Wrap model into a service (e.g., API endpoint).
- Output: deployable service

### 11. Model Monitoring
- Track:
  - predictions
  - performance over time
- Output: logs and monitoring metrics

### 12. Production Data Monitoring
- Detect:
  - data drift
  - distribution shifts
- Output: monitoring report

### 13. System Monitoring
- Monitor service health (latency, errors).
- Output: system logs and alerts

### 14. Pre-inference Transformations
- Apply transformations to incoming data before inference.
- Output: transformed input

### 15. Post-inference Transformations
- Apply:
  - output filtering
  - bias mitigation
  - privacy-preserving transformations
- Output: adjusted predictions

---

## IMPLEMENTATION GUIDELINES

- Use modular design: each operation should be callable independently.
- Use container-friendly code (no hard dependencies on local state).
- Prefer open-source libraries when relevant (e.g., sklearn, evidently, aif360).
- Ensure each operation logs:
  - inputs
  - outputs
  - execution status
- Provide example usage for each module.

---

## EXPECTED OUTPUT

- A structured codebase with:
  - one module per operation
  - shared artifact definitions
  - configuration files (YAML)
- Clear documentation explaining how operations connect into a pipeline.

---

## IMPORTANT

Do NOT implement everything in a single script.
Focus on:
- modularity
- reusability
- traceability between operations and requirements

Each operation must be independently testable and composable into a full AI pipeline.