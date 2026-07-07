# PF-Based Adaptive Distance Protection Dataset Generator

Generates structured datasets for training machine learning to predict adaptive distance-protection zone reach parameters in sub-transmission grids with distributed generation (DG).

The pipeline connects to a live **DIgSILENT PowerFactory** session, iterates over switch-state topology configurations and randomized line/DG scenarios, runs short-circuit calculations per DG turbine to compute in-feed corrections, and exports flat CSV rows plus graph-array Parquet files ready for ML training.

---

## Repository Structure

```text
pf-based-adaptive-distance-protection-dataset-generator/
├── main_script.py                        ← pipeline entry point
├── README.md
├── requirements.txt
├── .gitignore
├── inspect_grid_bus_types.py             ← standalone diagnostic: maps terminals to bus_typ codes
├── generate_switch_states.py             ← automated switch-state generator (no GUI needed)
├── create_manual_switch_state_library.py ← manual switch-state recorder via PowerFactory GUI
├── view_parquet.py                       ← diagnostic viewer for generated Parquet files
├── switch_state/
│   └── switch_states.csv                ← canonical switch-state library (93 states, version-controlled)
└── pf_adaptive_distance_dataset/
    ├── core/
    │   ├── config.py                     ← all user-configurable settings
    │   ├── models.py                     ← dataclasses (ExportPayload, DatasetStatistics, …)
    │   └── dataset_schema.py             ← flat column lists and graph array column groups
    ├── pf_api/
    │   ├── pf_session.py                 ← PowerFactory session management
    │   ├── pf_utils.py                   ← safe attribute access helpers
    │   ├── slave_cases.py                ← slave study-case / operation-scenario lifecycle
    │   ├── state_capture.py              ← captures original grid state + bus attributes
    │   └── grid_state.py                 ← restores grid state after each scenario
    ├── domain/
    │   ├── topology.py                   ← line/terminal/cubicle helpers
    │   ├── network_topology.py           ← corridor detection, branch summaries, parallel detection
    │   ├── dg_utils.py                   ← DG discovery and capacity reading
    │   ├── zone_reach.py                 ← Zone 1/2/3 reach formulas (simple + complex parallel)
    │   └── infeed.py                     ← short-circuit infeed correction per DG per zone
    ├── pipeline/
    │   ├── dataset_generator.py          ← main streaming generation loop (slave-case workflow)
    │   ├── switch_states.py              ← switch-state CSV loading and application
    │   ├── randomization.py              ← line-length and DG-capacity randomisation
    │   └── case_features.py              ← flat feature row builder per protected corridor
    ├── graph/
    │   ├── graph_arrays.py               ← converts scenario rows → graph-array row (Y-bus, edges, bus_typ)
    │   └── graph_array_utils.py          ← index helpers, upper-triangle, canonical IDs
    └── exports/
        ├── export.py                     ← streaming export: CSV, Parquet, Excel audit, JSON stats
        └── validation.py                 ← row-level validation and numeric statistics
```

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Required packages: `pandas`, `openpyxl`, `pyarrow`

PowerFactory must be installed separately. The pipeline uses its built-in Python API — no additional installation is needed beyond pointing `PF_PYTHON_PATH` at the correct folder (see Configuration below).

### Main Entry Point

```bash
python main_script.py
```

The script performs the full workflow:

1. Initialises the random seed
2. Loads switch-state configurations from `switch_state/switch_states.csv`
3. Opens a PowerFactory session and activates the configured project
4. For each switch-state row, creates a slave study-case / operation-scenario pair
5. Captures original grid state and bus-type attributes (terminal classifications)
6. For each scenario (base + randomised):
   - Applies the switch state and randomised line/DG parameters
   - Extracts flat corridor-level feature rows (Zone 1/2/3 reach + infeed corrections)
   - Builds a graph-array row (Y-bus, directed edges, node types, switch status)
   - Streams payloads to disk
7. Deletes each slave pair after processing
8. Writes final Parquet, audit Excel, metadata JSON, and statistics JSON

---

## Configuration

All user-configurable settings live in `pf_adaptive_distance_dataset/core/config.py`.

### PowerFactory Connection

```python
PF_PYTHON_PATH   = r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP3\Python\3.9"
PROJECT_NAME     = r"\your_user\your_folder\YourProject.IntPrj"
GRID_NAME        = "Grid_110kV.ElmNet"

MASTER_STUDY_CASE_NAME          = "SC_Master"
MASTER_OPERATION_SCENARIO_NAME  = "OS_Master"
```

Update these to match your local PowerFactory installation and project structure before running.

### Zone Reach Settings

```python
REACH_GF             = 0.85   # grading factor for Zone 1 and Zone 2
ZONE3_REACH_FACTOR   = 1.20   # multiplier on downstream branch impedance for Zone 3
```

### Dataset Volume Controls

```python
ENABLE_SWITCH_STATE_SCENARIOS   = True
MAX_SWITCH_STATE_CONFIG_COUNT   = None   # None = all rows; set e.g. 2 for a quick test run

INCLUDE_ORIGINAL_BASE_CASE      = True
RANDOMIZED_SCENARIO_COUNT       = 2      # randomised scenarios per switch state
```

Example with defaults and 86 switch states:
`86 switch states × (1 base + 2 randomised) = 258 scenario runs ≈ 4,200 flat rows`

### Randomisation Controls

```python
ENABLE_LINE_RANDOMIZATION      = True
LINE_LENGTH_SCALE_MIN          = 0.8    # scale line dline attribute down to 80 %
LINE_LENGTH_SCALE_MAX          = 1.2    # scale line dline attribute up to 120 %

ENABLE_DG_CAPACITY_RANDOMIZATION = True
DG_CAPACITY_SCALE_MIN          = 0.8    # scale installed DG capacity (sgn/Sn) down to 80 %
DG_CAPACITY_SCALE_MAX          = 1.2    # scale installed DG capacity (sgn/Sn) up to 120 %
```

Line randomisation varies the `dline` attribute only. PowerFactory recomputes R1/X1 from the shared line type, preserving the per-unit impedance of the original model.

### Reproducibility

```python
RANDOM_SEED_BASE: Optional[int] = None   # auto-generated from timestamp + PID
```

For repeatable runs, set a fixed integer:

```python
RANDOM_SEED_BASE: Optional[int] = 123456
```

The seed used in each run is logged and stored in the metadata JSON so any run can be reproduced exactly.

---

## Inputs and Outputs

### Switch-State Input

```text
switch_state/switch_states.csv
```

The canonical switch state library is version controlled at `switch_state/switch_states.csv`. It contains **86 switch states** covering:

- All 8 single-line outages
- All 28 double-line outage combinations
- 4 triple-line outages (extreme topology reductions)
- 24 states combining line outages with DG outages

Each row is one topology configuration. Columns named `switch_<CIM-RDF-ID>` contain `0` (open) or `1` (closed) for each switchable cubicle in the grid.

### Output Files

All outputs are written to `Results/` with a timestamp suffix.

