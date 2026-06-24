# Unified Graph-Array Distance Protection Dataset Generator

This repository contains a PowerFactory-based dataset generation pipeline for distance protection parameter prediction.

The main script, `distance-protection-dataset-generator.py`, connects to DIgSILENT PowerFactory, activates the configured 110 kV grid model, applies switch-state scenarios, optionally randomizes line lengths and distributed-generation capacities, calculates distance protection reach values, and exports both graph-array and flat audit datasets.

The generated dataset is intended for machine-learning workflows where grid topology, switched admittance matrices, relay corridor features, distributed-generation context, and Zone 1/2/3 relay reach targets are required.


## What the script does

The pipeline performs the following operations:

1. **Connects to DIgSILENT PowerFactory**

   * Opens PowerFactory through the Python API.
   * Activates the configured PowerFactory project.
   * Selects the configured grid object.
2. **Loads switch-state scenarios**

   * Reads `Switch_state.csv` from the expected `Results/Switch State/` folder.
   * Applies each switch-state configuration to PowerFactory cubicles.
   * Treats switch value `1` as closed and `0` as open.
3. **Updates topology according to open switches**

   * Detects objects connected behind open cubicle switches.
   * Sets affected lines, generators, loads, and transformers out of service for the active scenario.
4. **Generates scenario variants**

   * Supports the original base case.
   * Optionally generates randomized line-length scenarios.
   * Optionally generates randomized distributed-generation capacity scenarios.
   * Stores the random seed and scale range for traceability.
5. **Extracts protection-relevant grid features**

   * Detects directional protected corridors from relay-side busbars to subsequent busbars.
   * Identifies downstream branches, parallel corridors, junction nodes, and distributed-generation locations.
   * Calculates base Zone 1, Zone 2, and Zone 3 relay reach values.
   * Calculates in-feed correction terms using short-circuit simulations.
   * Exports corrected target reach values for machine-learning training.
6. **Builds graph-array topology data**

   * Generates bus lists and bus indices.
   * Builds directed edge arrays.
   * Builds physical edge arrays.
   * Stamps switched admittance matrix data.
   * Stores graph-level arrays in a `.paraquet` dataset.
7. **Exports audit and metadata files**

   * Streams graph rows directly to`.paraquet`.
   * Streams flat corridor rows to `.csv`.
   * Creates an `.xlsx` audit file.
   * Saves metadata, statistics, and randomization logs.

## Software requirements

Use **Python 3.9**, because the script is configured for **DIgSILENT PowerFactory 2023 SP3 Python 3.9**.

PowerFactory must be installed and licensed on the system where the script is executed. The `powerfactory` Python module is provided by the PowerFactory installation and is **not** installed from pip.

Recommended `requirements.txt`:

```text
pandas
pyarrow
```

## Installation

Clone the repository:

```bash
git clone https://github.com/aswinkrishnapoyil/pf-based-adaptive-distance-protection-dataset-generator.git
```

Create and activate a Python 3.9 virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install all requirements with pip

```bash
pip install -r requirements.txt
```

## PowerFactory configuration

Before running the script, check the `Config` class in `distance-protection-dataset-generator.py`.

```python
class Config:
    PF_PYTHON_PATH = r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP3\Python\3.9"
    PROJECT_NAME = r"\ic84yhos\Venus\VeN2uS_ExampleGrid_v1.4_FRT_KP"
    GRID_NAME = "Grid_110kV.ElmNet"
```

Update these values if the supervisor's system uses a different PowerFactory installation path, project path, or grid name.

|Setting|Meaning|
|-|-|
|`PF_PYTHON_PATH`|Folder containing the PowerFactory Python API module|
|`PROJECT_NAME`|Exact PowerFactory project path to activate|
|`GRID_NAME`|Exact grid object name inside the active project|
|`DATASET_VERSION`|Version label written to metadata|
|`REACH_GF`|Zone reach grading factor for Zone 1 and 2, currently `0.85`|
|`ZONE3_REACH_FACTOR`|Zone 3 downstream branch multiplier, currently `1.20`|

## Expected folder structure

The script defines paths relative to the location of `distance-protection-dataset-generator.py`.

```text
pf-based-adaptive-distance-protection-dataset-generator/
├── distance-protection-dataset-generator.py
├── README.md
├── requirements.txt
├── .gitignore
├── logs/
│   └── pipeline.log
└── Results/
    ├── Switch State/
    │   └── Switch_state.csv
    ├── case_feature_matrix_graph_array_topology.parquet
    ├── case_feature_matrix_randomized_grid_scenarios.csv
    ├── dataset_metadata.json
    └── dataset_statistics.json 
```

## Switch-state input file

The script expects the switch-state configuration file at:

```text
Results/Switch State/Switch_state.csv
```

Expected format:

```csv
ConfigID;switch_<cubicle_cimRdfId_1>;switch_<cubicle_cimRdfId_2>;switch_<cubicle_cimRdfId_3>
547e4bb8-9306-5355-89e5-a58215e9ed82;1;1;1
c004d72b-8523-54db-aa4c-ab5f045b59d4;1;0;1
```

## Scenario-generation settings

The most important settings are in the `Config` class.

```python
ENABLE_LINE_RANDOMIZATION = False
RANDOMIZED_SCENARIO_COUNT = 0
RANDOM_SEED_BASE: Optional[int] = None
LINE_LENGTH_SCALE_MIN = 0.8
LINE_LENGTH_SCALE_MAX = 1.2
INCLUDE_ORIGINAL_BASE_CASE = True

ENABLE_DG_CAPACITY_RANDOMIZATION = False
DG_CAPACITY_SCALE_MIN = 0.8
DG_CAPACITY_SCALE_MAX = 1.2
DG_CAPACITY_RANDOM_SEED_OFFSET = 100000

ENABLE_SWITCH_STATE_SCENARIOS = True
MAX_SWITCH_STATE_CONFIG_COUNT = 1
```

### Current default behavior

With the current default settings, the script runs a small smoke-test dataset:

* only the first switch-state configuration is used.
* only the original base case is generated.
* line randomization is disabled.
* distributed-generation capacity randomization is disabled.

This is suitable for testing whether the PowerFactory connection, switch-state mapping, corridor extraction, and file export work correctly.

### Recommended full dataset settings

For a large dataset generation run, update the settings deliberately.

Example:

```python
ENABLE_LINE_RANDOMIZATION = True
RANDOMIZED_SCENARIO_COUNT = 100
RANDOM_SEED_BASE: Optional[int] = None
LINE_LENGTH_SCALE_MIN = 0.8
LINE_LENGTH_SCALE_MAX = 1.2
INCLUDE_ORIGINAL_BASE_CASE = True

ENABLE_DG_CAPACITY_RANDOMIZATION = True
DG_CAPACITY_SCALE_MIN = 0.8
DG_CAPACITY_SCALE_MAX = 1.2
DG_CAPACITY_RANDOM_SEED_OFFSET = 100000

ENABLE_SWITCH_STATE_SCENARIOS = True
MAX_SWITCH_STATE_CONFIG_COUNT = None
```

Use a fixed `RANDOM_SEED_BASE` for reproducible datasets. Leave it as `None` only when a new seed should be generated automatically.

