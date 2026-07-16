# aequitas-service
Decompose aequitas into a modular production-ready service.
This repo contains functionalities that wrap aequitas framework and its components into an end-to-end service to embed compliance into every operation of the AI lifecycle.

## Prerequisites

Before using this service, ensure the following are installed:

- **Python 3.10**

- **templops** — [MLOps framework library](https://github.com/AlbanaCelepija/enhanced_mlops)

- **aequitas-fairlib** — [Fairness library
  Documentation](https://pikalab-unibo-students.github.io/master-thesis-dizio-ay2324/)

## Running the GUI

From the repository root:

1. Install the GUI dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install the `temlops` library from the `framework` folder:
   ```bash
   pip install -e framework
   ```

3. Set the `FAIROPS_ONTOLOGY_PATH` environment variable to point to the `fairops.ttl` file of the [fairops ontology](https://github.com/ai-unibo/fairops/tree/main/docs). The GUI reads this variable to populate the interface (Application Domain, AI Task, AI Type of Use, Fairness Concerns, Notions, Metrics and Mitigation Techniques) from the ontology. `indiv.ttl` must live in the same folder as `fairops.ttl`, since it is loaded from there automatically.

   Copy `.env.example` to `.env` and fill in the path:
   ```bash
   cp .env.example .env
   ```
   ```bash
   FAIROPS_ONTOLOGY_PATH=/absolute/path/to/fairops/docs/fairops.ttl
   ```

4. Launch the Streamlit app:
   ```bash
   streamlit run gui/app.py
   ```

