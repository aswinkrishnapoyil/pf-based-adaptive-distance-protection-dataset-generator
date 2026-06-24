# Unified Graph-Array Distance Protection Dataset Generator

This repository contains a PowerFactory-based dataset generation pipeline for distance protection parameter prediction.

The main script, `v2.0.py`, connects to DIgSILENT PowerFactory, activates the configured 110 kV grid model, applies switch-state scenarios, optionally randomizes line lengths and distributed-generation capacities, calculates distance protection reach values, and exports both graph-array and flat audit datasets.

The generated dataset is intended for machine-learning workflows where grid topology, switched admittance matrices, relay corridor features, distributed-generation context, and Zone 1/2/3 relay reach targets are required.

\---

## What the script does

The pipeline performs the following operations:

1. **Connects to DIgSILENT PowerFactory**

   * Opens PowerFactory through the Python API.
   * Activates the configured PowerFactory project.
   * Selects the configured grid object.
2. **Loads switch-state scenarios**

   * Reads `Switch\\\_state.csv` from the expected `Results/Switch State/` folder.
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

\---

## Software requirements

Use **Python 3.9**, because the script is configured for **DIgSILENT PowerFactory 2023 SP3 Python 3.9**.

PowerFactory must be installed and licensed on the system where the script is executed. The `powerfactory` Python module is provided by the PowerFactory installation and is **not** installed from pip.

Recommended `requirements.txt`:

```text
pandas
pyarrow
```

\---

## Installation

Clone the repository:

```bash
git clone https://github.com/aswinkrishnapoyil/pf-based-adaptive-distance-protection-dataset-generator.git
```

Create and activate a Python 3.9 virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\\\\.venv\\\\Scripts\\\\Activate.ps1
```

Install all requirements with pip

```bash
pip install -r requirements.txt
```

\---

## PowerFactory configuration

Before running the script, check the `Config` class in `v2.0.py`.

```python
class Config:
    PF\\\_PYTHON\\\_PATH = r"C:\\\\Program Files\\\\DIgSILENT\\\\PowerFactory 2023 SP3\\\\Python\\\\3.9"
    PROJECT\\\_NAME = r"\\\\ic84yhos\\\\Venus\\\\VeN2uS\\\_ExampleGrid\\\_v1.4\\\_FRT\\\_KP"
    GRID\\\_NAME = "Grid\\\_110kV.ElmNet"
```

Update these values if the supervisor's system uses a different PowerFactory installation path, project path, or grid name.

|Setting|Meaning|
|-|-|
|`PF\\\_PYTHON\\\_PATH`|Folder containing the PowerFactory Python API module|
|`PROJECT\\\_NAME`|Exact PowerFactory project path to activate|
|`GRID\\\_NAME`|Exact grid object name inside the active project|
|`DATASET\\\_VERSION`|Version label written to metadata|
|`REACH\\\_GF`|Zone reach grading factor for Zone 1 and 2, currently `0.85`|
|`ZONE3\\\_REACH\\\_FACTOR`|Zone 3 downstream branch multiplier, currently `1.20`|

> Note: the script file is named `v2.0.py`, but the internal `DATASET\\\_VERSION` value should also be checked before a final run. If the exported metadata should say `v2.0`, update `DATASET\\\_VERSION` accordingly.

\---

## Expected folder structure

The script defines paths relative to the location of `v2.0.py`.

```text
distance-protection-parameter-prediction/
│
├── Results/
│   └── Switch State/
│       └── Switch\\\_state.csv
│
├── dataset\\\_generator/
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
SCRIPT\\\_DIR = Path(\\\_\\\_file\\\_\\\_).resolve().parent
PROJECT\\\_ROOT = SCRIPT\\\_DIR.parent
OUTPUT\\\_DIR = SCRIPT\\\_DIR
RESULTS\\\_DIR = PROJECT\\\_ROOT / "Results"
SWITCH\\\_STATE\\\_FILE = RESULTS\\\_DIR / "Switch State" / "Switch\\\_state.csv"
```

This means:

* output files are written into the same folder as `v2.0.py`;
* `Switch\\\_state.csv` is expected one directory above the script folder, under `Results/Switch State/`;
* if `v2.0.py` is moved, the location of `Switch\\\_state.csv` changes accordingly.

\---

## Switch-state input file

The script expects the switch-state configuration file at:

```text
Results/Switch State/Switch\\\_state.csv
```

Expected format:

```csv
ConfigID;switch\\\_<cubicle\\\_cimRdfId\\\_1>;switch\\\_<cubicle\\\_cimRdfId\\\_2>;switch\\\_<cubicle\\\_cimRdfId\\\_3>
547e4bb8-9306-5355-89e5-a58215e9ed82;1;1;1
c004d72b-8523-54db-aa4c-ab5f045b59d4;1;0;1
```

Rules:

* `ConfigID` identifies the switch-state configuration.
* Each switch-state column must start with `switch\\\_`.
* The part after `switch\\\_` must match the corresponding PowerFactory cubicle `cimRdfId`.
* Leading underscores in the PowerFactory `cimRdfId` are ignored by the script.
* `1` means the switch is closed.
* `0` means the switch is open.
* Blank or non-numeric switch values are interpreted as `1`, meaning closed.
* Any column that does not start with `switch\\\_` is ignored for switch-state application.

> Note: the script reads the `.csv` file with automatic delimiter detection. However, the provided `Switch\\\_state.csv` uses semicolons, so the recommended format is semicolon-separated. 
If `MAX\\\_SWITCH\\\_STATE\\\_CONFIG\\\_COUNT` is set in `Config`, only the first selected number of switch-state rows are processed..

\---

## Scenario-generation settings

The most important settings are in the `Config` class.

```python
ENABLE\\\_LINE\\\_RANDOMIZATION = False
RANDOMIZED\\\_SCENARIO\\\_COUNT = 0
RANDOM\\\_SEED\\\_BASE: Optional\\\[int] = None
LINE\\\_LENGTH\\\_SCALE\\\_MIN = 0.8
LINE\\\_LENGTH\\\_SCALE\\\_MAX = 1.2
INCLUDE\\\_ORIGINAL\\\_BASE\\\_CASE = True

ENABLE\\\_DG\\\_CAPACITY\\\_RANDOMIZATION = False
DG\\\_CAPACITY\\\_SCALE\\\_MIN = 0.8
DG\\\_CAPACITY\\\_SCALE\\\_MAX = 1.2
DG\\\_CAPACITY\\\_RANDOM\\\_SEED\\\_OFFSET = 100000

ENABLE\\\_SWITCH\\\_STATE\\\_SCENARIOS = True
MAX\\\_SWITCH\\\_STATE\\\_CONFIG\\\_COUNT = 1
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
ENABLE\\\_LINE\\\_RANDOMIZATION = True
RANDOMIZED\\\_SCENARIO\\\_COUNT = 100
RANDOM\\\_SEED\\\_BASE: Optional\\\[int] = None
LINE\\\_LENGTH\\\_SCALE\\\_MIN = 0.8
LINE\\\_LENGTH\\\_SCALE\\\_MAX = 1.2
INCLUDE\\\_ORIGINAL\\\_BASE\\\_CASE = True

ENABLE\\\_DG\\\_CAPACITY\\\_RANDOMIZATION = True
DG\\\_CAPACITY\\\_SCALE\\\_MIN = 0.8
DG\\\_CAPACITY\\\_SCALE\\\_MAX = 1.2
DG\\\_CAPACITY\\\_RANDOM\\\_SEED\\\_OFFSET = 100000

ENABLE\\\_SWITCH\\\_STATE\\\_SCENARIOS = True
MAX\\\_SWITCH\\\_STATE\\\_CONFIG\\\_COUNT = None
```

Use a fixed `RANDOM\\\_SEED\\\_BASE` for reproducible datasets. Leave it as `None` only when a new seed should be generated automatically.

