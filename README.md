As an AI, I don't have the ability to send direct file downloads (like a `.zip` or a `.md` file attachment) to your computer.

However, getting the file is quick and easy! Just click the **"Copy code"** button in the top right corner of the block below, paste it into your editor (like VS Code, Notepad, or PyCharm), and save it exactly as `README.md`.

```markdown
# PF-Based Adaptive Distance Protection Dataset Generator

This repository contains a modular Python pipeline for generating adaptive distance-protection datasets from DIgSILENT PowerFactory. 

The pipeline creates switch-state and randomized operating scenarios, extracts distance-protection corridor features, calculates protection-zone reaches, evaluates distributed-generation infeed effects, and exports both flat tabular data and graph-array style data for machine-learning workflows.

---

## 📑 Table of Contents
- [Project Purpose](#project-purpose)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Main Entry Point](#main-entry-point)
- [Configuration](#configuration)
  - [PowerFactory Settings](#powerfactory-settings)
  - [Scenario Configuration](#scenario-configuration)
- [Inputs and Outputs](#inputs-and-outputs)
  - [Switch-State Input](#switch-state-input)
  - [Output Files](#output-files)
  - [Dataset Content](#dataset-content)
- [Package Overview](#package-overview)
- [Development Checks](#development-checks)
- [Reproducibility](#reproducibility)
- [Notes and Limitations](#notes-and-limitations)
- [License](#license)

---

## 🎯 Project Purpose

The goal of this project is to generate structured datasets for predicting or analyzing adaptive distance-protection parameters in distribution or sub-transmission grids with distributed generation (DG).

**The generated dataset includes:**
* Protected corridor topology features
* Zone 1, Zone 2, and Zone 3 distance reach values
* Downstream branch and parallel-corridor information
* Distributed generation location and capacity features
* Short-circuit based DG infeed correction features
* Switch-state scenario metadata
* Randomized line-length and DG-capacity scenario metadata
* Graph-array representations of the network scenario

---

## 📂 Repository Structure

```text
pf-based-adaptive-distance-protection-dataset-generator/
│
├── main_script.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── pf_adaptive_distance_dataset/
    │
    ├── core/
    │   ├── config.py
    │   ├── models.py
    │   └── dataset_schema.py
    │
    ├── pf_api/
    │   ├── pf_session.py
    │   ├── pf_utils.py
    │   ├── slave_cases.py
    │   ├── state_capture.py
    │   └── grid_state.py
    │
    ├── domain/
    │   ├── topology.py
    │   ├── network_topology.py
    │   ├── dg_utils.py
    │   ├── zone_reach.py
    │   └── infeed.py
    │
    ├── pipeline/
    │   ├── dataset_generator.py
    │   ├── switch_states.py
    │   ├── randomization.py
    │   └── case_features.py
    │
    ├── graph/
    │   ├── graph_arrays.py
    │   └── graph_array_utils.py
    │
    └── exports/
        ├── export.py
        └── validation.py

```

---

## 🚀 Getting Started

### Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt

```

*Required packages:* `pandas`, `openpyxl`, `pyarrow`

> **Note:** The `powerfactory` Python module is not installed via pip. It is provided by your local DIgSILENT PowerFactory installation.

### Main Entry Point

Run the pipeline from the repository root:

```bash
python main_script.py

```

**The main script performs the full workflow:**

1. Loads switch-state configurations
2. Opens a DIgSILENT PowerFactory session
3. Creates slave study cases and slave operation scenarios
4. Applies switch states and randomized line/DG scenarios
5. Extracts flat corridor-level features
6. Builds graph-array scenario rows
7. Exports dataset files and audit outputs

---

## ⚙️ Configuration

### PowerFactory Settings

PowerFactory-specific settings are defined in `pf_adaptive_distance_dataset/core/config.py`. Important configuration values include:

```python
PF_PYTHON_PATH = r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP3\Python\3.9"
PROJECT_NAME = r"\ic84yhos\Venus\VeN2uS_ExampleGrid_v1.4_FRT_KP"
GRID_NAME = "Grid_110kV.ElmNet"

MASTER_STUDY_CASE_NAME = "Study Case"
MASTER_OPERATION_SCENARIO_NAME = "OS_Master"

```

The pipeline expects a master study case and master operation scenario to exist. For each switch-state configuration, the pipeline creates a slave study case/scenario pair, processes all scenarios, and then safely deletes the slaves.

### Scenario Configuration

Scenario controls are also located in `config.py`:

```python
ENABLE_SWITCH_STATE_SCENARIOS = True
MAX_SWITCH_STATE_CONFIG_COUNT = 2

INCLUDE_ORIGINAL_BASE_CASE = True
RANDOMIZED_SCENARIO_COUNT = 1

ENABLE_LINE_RANDOMIZATION = True
LINE_LENGTH_SCALE_MIN = 0.8
LINE_LENGTH_SCALE_MAX = 1.2

ENABLE_DG_CAPACITY_RANDOMIZATION = True
DG_CAPACITY_SCALE_MIN = 0.8
DG_CAPACITY_SCALE_MAX = 1.2

```

*Example runtime with the above config: `2 switch states × (1 base scenario + 1 randomized scenario) = 4 scenario runs*`

---

## 📊 Inputs and Outputs

### Switch-State Input

Switch-state configurations are read from:

```text
Results/Switch State/Switch_state.csv

```

Switch-state columns must use the prefix `switch_` (e.g., `ConfigID,switch_<rdf_id_1>,switch_<rdf_id_2>`).

Each switch value is normalized as:

* `1` = closed
* `0` = open

*(If switch-state scenarios are disabled, the pipeline defaults to the live grid state).*

### Output Files

Generated outputs are written to the `Results/` directory. Typical outputs include:

* Flat CSV rows for corridor-level machine-learning features
* Graph-array scenario rows (`.parquet`)
* Excel audit files (`.xlsx`)
* Randomization logs
* Dataset statistics and metadata JSON files

> *Generated outputs and logs are intentionally ignored by Git.*

### Dataset Content

**The flat dataset contains:**

* Relay and corridor identifiers
* Protected corridor length, resistance, and reactance
* Parallel corridor indicators and downstream branch features
* Zone 1, Zone 2, and Zone 3 reach values
* DG counts, capacities, and infeed correction values
* Final target reach values

**The graph-array dataset contains:**

* Bus/node identifiers
* Directed edge and physical edge arrays
* Switched Y-bus arrays
* Switch-state vectors and scenario metadata

---

## 📦 Package Overview

* **`core`**: Configuration, dataclasses, and dataset schema definitions.
* **`pf_api`**: PowerFactory session handling, safe attribute access, slave-case management, grid-state capture, and state restoration utilities.
* **`domain`**: Topology extraction, protected-corridor detection, DG summaries, zone-reach calculations, and short-circuit based infeed correction logic.
* **`pipeline`**: Main dataset generation workflow, switch-state handling, scenario randomization, and flat feature extraction.
* **`graph`**: Utilities for converting scenario rows into graph-array representations.
* **`exports`**: Streaming export, validation, statistics, and audit writing logic.

---

## 🛠 Development Checks

Before committing changes, ensure your code compiles correctly:

```bash
python -m py_compile main_script.py
python -m compileall pf_adaptive_distance_dataset
python -c "import main_script; print('main_script import ok')"

```

*Expected output: `main_script import ok*`

**Git Notes:**
Generated files should not be committed. The `.gitignore` excludes `Results/`, `logs/`, `__pycache__/`, `*.parquet`, `*.xlsx`, `*.log`, as well as temporary PF files.

---

## 🔄 Reproducibility

By default, the random seed is auto-generated based on timestamp and process ID:

```python
RANDOM_SEED_BASE = None

```

For repeatable dataset generation, set a fixed integer value in `config.py`:

```python
RANDOM_SEED_BASE = 123456

```

---

## ⚠️ Notes and Limitations

* Requires access to a local DIgSILENT PowerFactory installation.
* The `powerfactory` Python module must be available through the configured Python path.
* Target object names (study cases, scenarios, grids) must exactly match the active project.
* Generated datasets depend on the active PowerFactory model, the switch-state CSV, and randomization settings.
* Designed for controlled research and dataset-generation workflows, **not** for direct real-time protection operation.

---

## 📄 License

*(Add license information here before public release).*

```

If you meant you wanted a script to automatically download/build the entire 22-file directory structure on your machine, let me know! I can write a Python script that will instantly scaffold the whole project for you.

```
