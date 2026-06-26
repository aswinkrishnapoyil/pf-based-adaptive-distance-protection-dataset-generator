# PF-Based Adaptive Distance Protection Dataset Generator

The goal of this project is to generate structured datasets for predicting or analyzing adaptive distance-protection parameters in sub-transmission grids with distributed generation (DG).

## Repository Structure

```text
pf-based-adaptive-distance-protection-dataset-generator/
├── main_script.py
├── README.md
├── requirements.txt
├── .gitignore
└── pf_adaptive_distance_dataset/
    ├── core/
    │   ├── config.py
    │   ├── models.py
    │   └── dataset_schema.py
    ├── pf_api/
    │   ├── pf_session.py
    │   ├── pf_utils.py
    │   ├── slave_cases.py
    │   ├── state_capture.py
    │   └── grid_state.py
    ├── domain/
    │   ├── topology.py
    │   ├── network_topology.py
    │   ├── dg_utils.py
    │   ├── zone_reach.py
    │   └── infeed.py
    ├── pipeline/
    │   ├── dataset_generator.py
    │   ├── switch_states.py
    │   ├── randomization.py
    │   └── case_features.py
    ├── graph/
    │   ├── graph_arrays.py
    │   └── graph_array_utils.py
    └── exports/
        ├── export.py
        └── validation.py
```

## Getting Started

### Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt

```

*Required packages:* `pandas`, `openpyxl`, `pyarrow`

### Main Entry Point

Run the pipeline from the repository root:

```bash
python main_script.py

```

**The main script performs the full workflow:**

1. Loads switch-state configurations
2. Opens a PowerFactory session
3. Creates slave study cases and slave operation scenarios
4. Applies switch states and randomized line/DG scenarios
5. Extracts flat corridor-level features
6. Builds graph-array scenario rows
7. Exports dataset files and audit outputs

## Configuration

### PowerFactory Settings
PowerFactory-specific settings are defined in `pf_adaptive_distance_dataset/core/config.py`. **To run this pipeline locally, you must update these values to match your specific PowerFactory installation and project environment.**

```python
# UPDATE THESE TO MATCH YOUR LOCAL SETUP:
PF_PYTHON_PATH = r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP3\Python\3.9" # Path to your PF Python API folder
PROJECT_NAME = r"\Your\Project\Path\Here" # The exact path to your project inside PowerFactory
GRID_NAME = "Grid_110kV.ElmNet" # The specific grid network you want to activate

MASTER_STUDY_CASE_NAME = "Study Case" # Name of your Study Case
MASTER_OPERATION_SCENARIO_NAME = "OS_Master" # Name of your Operation Scenario
```

For each switch-state configuration, the pipeline creates a slave study case/scenario pair, processes all scenarios, and then safely deletes the slaves.

### Scenario Configuration
Scenario controls dictate how many dataset rows are generated and how much the grid topology varies. These are also located in `config.py`.

You can adjust these to run quick tests or to generate massive datasets:

```python
# --- Switch State Controls ---
ENABLE_SWITCH_STATE_SCENARIOS = True
MAX_SWITCH_STATE_CONFIG_COUNT = 2  # Set to a low number (e.g., 2) for testing, or None to process all rows in CSV

# --- Base Case & Randomization Volume ---
INCLUDE_ORIGINAL_BASE_CASE = True  # Always include the un-modified grid state
RANDOMIZED_SCENARIO_COUNT = 1      # Number of randomized variations to generate PER switch state

# --- Line Length Randomization ---
ENABLE_LINE_RANDOMIZATION = True
LINE_LENGTH_SCALE_MIN = 0.8        # Scales line length down to 80%
LINE_LENGTH_SCALE_MAX = 1.2        # Scales line length up to 120%

# --- Distributed Generation (DG) Randomization ---
ENABLE_DG_CAPACITY_RANDOMIZATION = True
DG_CAPACITY_SCALE_MIN = 0.8        # Scales DG capacity down to 80%
DG_CAPACITY_SCALE_MAX = 1.2        # Scales DG capacity up to 120%
```

*Example runtime with the above config: 
`2 switch states × (1 base scenario + 1 randomized scenario) = 4 scenario runs*`

## Inputs and Outputs

### Switch-State Input

Switch-state configurations are read from:

```text
Results/Switch State/Switch_state.csv
```

### Output Files

Generated outputs are written to the `Results/` directory. Typical outputs include:

* Flat `.csv` rows for corridor-level features
* Graph-array scenario rows `.parquet`
* `.xlsx` audit files
* Randomization logs
* Dataset statistics and metadata `.json` files

## Package Overview

* **`core`**: Configuration, dataclasses, and dataset schema definitions.
* **`pf_api`**: PowerFactory session handling, safe attribute access, slave-case management, grid-state capture, and state restoration utilities.
* **`domain`**: Topology extraction, protected-corridor detection, DG summaries, zone-reach calculations, and short-circuit based infeed correction logic.
* **`pipeline`**: Main dataset generation workflow, switch-state handling, scenario randomization, and flat feature extraction.
* **`graph`**: Utilities for converting scenario rows into graph-array representations.
* **`exports`**: Streaming export, validation, statistics, and audit writing logic.

## Reproducibility

By default, the random seed is auto-generated based on timestamp and process ID:

```python
RANDOM_SEED_BASE: Optional[int] = None

```

For repeatable dataset generation, set a fixed integer value in `config.py`:

```python
RANDOM_SEED_BASE: Optional[int] = 123456

```
