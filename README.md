# Unified Graph-Array Distance Protection Dataset Generator

This repository contains a PowerFactory-based dataset generation pipeline for distance protection parameter prediction.

The main script, `v2.0.py`, connects to DIgSILENT PowerFactory, activates the configured 110 kV grid model, applies switch-state scenarios, optionally randomizes line lengths and distributed-generation capacities, calculates distance protection reach values, and exports both graph-array and flat audit datasets.

The generated dataset is intended for machine-learning workflows where grid topology, switched admittance matrices, relay corridor features, distributed-generation context, and Zone 1/2/3 relay reach targets are required.

---

## What the script does

The pipeline performs the following operations:

1. **Connects to DIgSILENT PowerFactory**
   - Opens PowerFactory through the Python API.
   - Activates the configured PowerFactory project.
   - Selects the configured grid object.

2. **Loads switch-state scenarios**
   - Reads `Switch_state.csv` from the expected `Results/Switch State/` folder.
   - Applies each switch-state configuration to PowerFactory cubicles.
   - Treats switch value `1` as closed and `0` as open.

3. **Updates topology according to open switches**
   - Detects objects connected behind open cubicle switches.
   - Sets affected lines, generators, loads, and transformers out of service for the active scenario.

4. **Generates scenario variants**
   - Supports the original base case.
   - Optionally generates randomized line-length scenarios.
   - Optionally generates randomized distributed-generation capacity scenarios.
   - Stores the random seed and scale range for traceability.

5. **Extracts protection-relevant grid features**
   - Detects directional protected corridors from relay-side busbars to subsequent busbars.
   - Identifies downstream branches, parallel corridors, junction nodes, and distributed-generation locations.
   - Calculates base Zone 1, Zone 2, and Zone 3 relay reach values.
   - Calculates in-feed correction terms using short-circuit simulations.
   - Exports corrected target reach values for machine-learning training.

6. **Builds graph-array topology data**
   - Generates bus lists and bus indices.
   - Builds directed edge arrays.
   - Builds physical edge arrays.
   - Stamps switched admittance matrix data.
   - Stores graph-level arrays in a `.paraquet` dataset.

7. **Exports audit and metadata files**
   - Streams graph rows directly to`.paraquet`.
   - Streams flat corridor rows to `.csv`.
   - Creates an `.xlsx` audit file.
   - Saves metadata, statistics, and randomization logs.

---

## Software requirements

Use **Python 3.9**, because the script is configured for **DIgSILENT PowerFactory 2023 SP3 Python 3.9**.

PowerFactory must be installed and licensed on the system where the script is executed. The `powerfactory` Python module is provided by the PowerFactory installation and is **not** installed from pip.

Recommended `requirements.txt`:

```text
pandas
pyarrow
openpyxl
```

---

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

---

## PowerFactory configuration

Before running the script, check the `Config` class in `v2.0.py`.

```python
class Config:
    PF_PYTHON_PATH = r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP3\Python\3.9"
    PROJECT_NAME = r"\ic84yhos\Venus\VeN2uS_ExampleGrid_v1.4_FRT_KP"
    GRID_NAME = "Grid_110kV.ElmNet"
```

Update these values if the supervisor's system uses a different PowerFactory installation path, project path, or grid name.

| Setting | Meaning |
|---|---|
| `PF_PYTHON_PATH` | Folder containing the PowerFactory Python API module |
| `PROJECT_NAME` | Exact PowerFactory project path to activate |
| `GRID_NAME` | Exact grid object name inside the active project |
| `DATASET_VERSION` | Version label written to metadata |
| `REACH_GF` | Zone reach grading factor for Zone 1 and 2, currently `0.85` |
| `ZONE3_REACH_FACTOR` | Zone 3 downstream branch multiplier, currently `1.20` |

> Note: the script file is named `v2.0.py`, but the internal `DATASET_VERSION` value should also be checked before a final run. If the exported metadata should say `v2.0`, update `DATASET_VERSION` accordingly.

---

## Expected folder structure

The script defines paths relative to the location of `v2.0.py`.

```text
distance-protection-parameter-prediction/
│
├── Results/
│   └── Switch State/
│       └── Switch_state.csv
│
├── dataset_generator/
│   ├── v2.0.py
│   ├── requirements.txt
│   ├── README.md
│   └── logs/
│       └── pipeline.log
│
└── .gitignore
```

Important path logic inside the script:

```python
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR
RESULTS_DIR = PROJECT_ROOT / "Results"
SWITCH_STATE_FILE = RESULTS_DIR / "Switch State" / "Switch_state.csv"
```

This means:

- output files are written into the same folder as `v2.0.py`;
- `Switch_state.csv` is expected one directory above the script folder, under `Results/Switch State/`;
- if `v2.0.py` is moved, the location of `Switch_state.csv` changes accordingly.

---

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

Rules:

- `ConfigID` identifies the switch-state configuration.
- Each switch-state column must start with `switch_`.
- The part after `switch_` must match the corresponding PowerFactory cubicle `cimRdfId`.
- Leading underscores in the PowerFactory `cimRdfId` are ignored by the script.
- `1` means the switch is closed.
- `0` means the switch is open.
- Blank or non-numeric switch values are interpreted as `1`, meaning closed.
- Any column that does not start with `switch_` is ignored for switch-state application.

> Note: the script reads the `.csv` file with automatic delimiter detection. However, the provided `Switch_state.csv` uses semicolons, so the recommended format is semicolon-separated. 
If `MAX_SWITCH_STATE_CONFIG_COUNT` is set in `Config`, only the first selected number of switch-state rows are processed..
---

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

- only the first switch-state configuration is used.
- only the original base case is generated.
- line randomization is disabled.
- distributed-generation capacity randomization is disabled.

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

---

## Dataset size estimate

The number of exported scenarios is approximately:

```text
number_of_switch_states × number_of_scenario_variants
```

where:

```text
number_of_scenario_variants = RANDOMIZED_SCENARIO_COUNT + 1
```

if `INCLUDE_ORIGINAL_BASE_CASE = True`.

The flat `.csv/.xlsx` row count is approximately:

```text
number_of_switch_states × number_of_scenario_variants × number_of_directional_protected_corridors
```

The `.parquet` row count is approximately:

```text
number_of_switch_states × number_of_scenario_variants
```

because the `.parquet` export stores one graph-array row per complete scenario.

---

## Running the generator

From the folder containing `v2.0.py`, run:

```bash
python v2.0.py
```

On Windows PowerShell:

```powershell
python .\v2.0.py
```

The script will open/connect to PowerFactory, activate the configured project, process the selected switch-state scenarios, and write the dataset outputs to disk.

---

## Output files

The following files are generated in the same folder as `v2.0.py`.

| File | Description |
|---|---|
| `case_feature_matrix_graph_array_topology.parquet` | Main graph-array dataset with one row per full grid scenario |
| `case_feature_matrix_randomized_grid_scenarios.csv` | Flat corridor-level audit dataset |
| `case_feature_matrix_randomized_grid_scenarios.xlsx` | Excel version of the flat audit dataset |
| `dataset_metadata.json` | Dataset metadata, including project name, grid name, timestamp, seed, and randomization settings |
| `dataset_statistics.json` | Validation counts, missing-value counts, numeric statistics, and reach-value ranges |
| `line_length_randomization_log.csv` | Line-length scale factors. Created only when line randomization is enabled |
| `dg_capacity_randomization_log.csv` | DG capacity scale factors. Created only when DG randomization is enabled |
| `logs/pipeline.log` | Full execution log |

---

## Main Parquet content

The Parquet dataset contains graph-level arrays and metadata for each scenario.

Important groups include:

| Column group | Description |
|---|---|
| Scenario metadata | Switch-state ID, scenario ID, random seeds, scale ranges |
| Switch vector | Binary switch-state vector and open/closed switch counts |
| Bus arrays | Bus IDs, bus indices, bus type placeholders |
| Directed edge arrays | Relay-side node, subsequent node, protected corridor ID, impedance, length, parallel information |
| Physical edge arrays | Unique physical graph edges used to stamp admittance data |
| Y-bus arrays | Switched admittance matrix values and connectivity arrays |
| Context features | Downstream branch context, DG context, turbine/infeed context, zone selection method fields |
| Base targets | Base Zone 1/2/3 R/X reach values |
| Correction terms | Zone 1 and Zone 2 infeed correction R/X terms |
| Final targets | Corrected target Zone 1/2/3 R/X reach values |

Target columns:

```text
target_zone1_r_reach_ohm
target_zone1_x_reach_ohm
target_zone2_r_reach_ohm
target_zone2_x_reach_ohm
target_zone3_r_reach_ohm
target_zone3_x_reach_ohm
```

---

## Flat CSV/XLSX audit content

The flat audit files contain one row per protected corridor per scenario.

These files are mainly for:

- manual inspection
- debugging scenario generation
- checking zone reach values
- checking downstream branch selection
- checking distributed-generation and infeed-correction behavior
- validating whether randomization behaves as expected

For machine-learning training, the `.parquet` graph-array file is the primary export.

---

## Validation and statistics

Each flat row is validated before being written to the `.csv` file.

The validation checks include:

- excessive missing numerical values
- unrealistic zone reach values
- strongly negative impedance values where they should not occur

Validation results are written to:

```text
dataset_statistics.json
```

The statistics file contains:

- total generated cases
- valid and invalid case counts
- invalid case reasons
- missing-value counts
- numeric summary statistics
- Zone reach ranges
- total generation time