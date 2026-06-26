\# PF-Based Adaptive Distance Protection Dataset Generator



This repository contains a modular Python pipeline for generating adaptive distance-protection datasets from DIgSILENT PowerFactory.



The pipeline creates switch-state and randomized operating scenarios, extracts distance-protection corridor features, calculates protection-zone reaches, evaluates distributed-generation infeed effects, and exports both flat tabular data and graph-array style data for machine-learning workflows.



\## Project Purpose



The goal of this project is to generate structured datasets for predicting or analysing adaptive distance-protection parameters in distribution or sub-transmission grids with distributed generation.



The generated dataset includes:



\- protected corridor topology features

\- Zone 1, Zone 2, and Zone 3 distance reach values

\- downstream branch and parallel-corridor information

\- distributed generation location and capacity features

\- short-circuit based DG infeed correction features

\- switch-state scenario metadata

\- randomized line-length and DG-capacity scenario metadata

\- graph-array representations of the network scenario



\## Repository Structure



```text

pf-based-adaptive-distance-protection-dataset-generator/

│

├── main\_script.py

├── README.md

├── requirements.txt

├── .gitignore

│

└── pf\_adaptive\_distance\_dataset/

&#x20;   │

&#x20;   ├── core/

&#x20;   │   ├── config.py

&#x20;   │   ├── models.py

&#x20;   │   └── dataset\_schema.py

&#x20;   │

&#x20;   ├── pf\_api/

&#x20;   │   ├── pf\_session.py

&#x20;   │   ├── pf\_utils.py

&#x20;   │   ├── slave\_cases.py

&#x20;   │   ├── state\_capture.py

&#x20;   │   └── grid\_state.py

&#x20;   │

&#x20;   ├── domain/

&#x20;   │   ├── topology.py

&#x20;   │   ├── network\_topology.py

&#x20;   │   ├── dg\_utils.py

&#x20;   │   ├── zone\_reach.py

&#x20;   │   └── infeed.py

&#x20;   │

&#x20;   ├── pipeline/

&#x20;   │   ├── dataset\_generator.py

&#x20;   │   ├── switch\_states.py

&#x20;   │   ├── randomization.py

&#x20;   │   └── case\_features.py

&#x20;   │

&#x20;   ├── graph/

&#x20;   │   ├── graph\_arrays.py

&#x20;   │   └── graph\_array\_utils.py

&#x20;   │

&#x20;   └── exports/

&#x20;       ├── export.py

&#x20;       └── validation.py

```



\## Main Entry Point



Run the pipeline from the repository root:



```bash

python main\_script.py

```



The main script performs the full workflow:



1\. loads switch-state configurations,

2\. opens a DIgSILENT PowerFactory session,

3\. creates slave study cases and slave operation scenarios,

4\. applies switch states,

5\. applies randomized line and DG scenarios,

6\. extracts flat corridor-level features,

7\. builds graph-array scenario rows,

8\. exports dataset files and audit outputs.



\## Requirements



Install the Python dependencies with:



```bash

pip install -r requirements.txt

```



The required pip packages are:



```text

pandas

openpyxl

pyarrow

```



The `powerfactory` Python module is not installed from pip. It is provided by the local DIgSILENT PowerFactory installation.



\## PowerFactory Configuration



The PowerFactory-specific settings are defined in:



```text

pf\_adaptive\_distance\_dataset/core/config.py

```



Important configuration values include:



```python

PF\_PYTHON\_PATH = r"C:\\Program Files\\DIgSILENT\\PowerFactory 2023 SP3\\Python\\3.9"

PROJECT\_NAME = r"\\ic84yhos\\Venus\\VeN2uS\_ExampleGrid\_v1.4\_FRT\_KP"

GRID\_NAME = "Grid\_110kV.ElmNet"



MASTER\_STUDY\_CASE\_NAME = "Study Case"

MASTER\_OPERATION\_SCENARIO\_NAME = "OS\_Master"

```



The pipeline expects a master study case and master operation scenario to exist in the PowerFactory project. For each switch-state configuration, the pipeline creates a slave study case and slave operation scenario, processes all scenarios, and then deletes the slave pair.



\## Scenario Configuration



The main scenario controls are also located in `config.py`.



Example:



```python

ENABLE\_SWITCH\_STATE\_SCENARIOS = True

MAX\_SWITCH\_STATE\_CONFIG\_COUNT = 2



INCLUDE\_ORIGINAL\_BASE\_CASE = True

RANDOMIZED\_SCENARIO\_COUNT = 1



ENABLE\_LINE\_RANDOMIZATION = True

LINE\_LENGTH\_SCALE\_MIN = 0.8

LINE\_LENGTH\_SCALE\_MAX = 1.2



ENABLE\_DG\_CAPACITY\_RANDOMIZATION = True

DG\_CAPACITY\_SCALE\_MIN = 0.8

DG\_CAPACITY\_SCALE\_MAX = 1.2

```



With this configuration:



```text

2 switch states × (1 base scenario + 1 randomized scenario) = 4 scenario runs

```



\## Switch-State Input



Switch-state configurations are read from:



```text

Results/Switch State/Switch\_state.csv

```



Switch-state columns should use the prefix:



```text

switch\_

```



For example:



```text

ConfigID,switch\_<rdf\_id\_1>,switch\_<rdf\_id\_2>,switch\_<rdf\_id\_3>

```



Each switch value is normalized as:



```text

1 = closed

0 = open

```



If switch-state scenarios are disabled, the pipeline uses the live grid state.



\## Output Files



Generated outputs are written under:



```text

Results/

```



Typical outputs include:



\- flat CSV rows for corridor-level machine-learning features

\- graph-array scenario rows

\- Excel audit files

\- randomization logs

\- dataset statistics

\- metadata JSON files



Generated outputs are intentionally ignored by Git.



\## Dataset Content



The flat dataset contains feature groups such as:



\- relay and corridor identifiers

\- protected corridor length, resistance, and reactance

\- parallel corridor indicators

\- downstream branch features

\- Zone 1, Zone 2, and Zone 3 reach values

\- distributed generation counts and capacities

\- DG infeed correction values

\- final target reach values



The graph-array dataset contains scenario-level arrays such as:



\- bus/node identifiers

\- directed edge arrays

\- physical edge arrays

\- switched Y-bus arrays

\- switch-state vectors

\- scenario metadata



\## Package Overview



\### `core`



Contains configuration, dataclasses, and dataset schema definitions.



\### `pf\_api`



Contains PowerFactory session handling, safe attribute access, slave-case management, grid-state capture, and state restoration utilities.



\### `domain`



Contains topology extraction, protected-corridor detection, DG summaries, zone-reach calculations, and short-circuit based infeed correction logic.



\### `pipeline`



Contains the main dataset generation workflow, switch-state handling, scenario randomization, and flat feature extraction.



\### `graph`



Contains utilities for converting scenario rows into graph-array representations.



\### `exports`



Contains streaming export, validation, statistics, and audit writing logic.



\## Development Checks



Before committing changes, run:



```bash

python -m py\_compile main\_script.py

python -m compileall pf\_adaptive\_distance\_dataset

python -c "import main\_script; print('main\_script import ok')"

```



Expected output:



```text

main\_script import ok

```



\## Git Notes



Generated files should not be committed. The `.gitignore` excludes:



```text

Results/

logs/

\_\_pycache\_\_/

\*.parquet

\*.xlsx

\*.log

```



PowerFactory project files and local temporary files are also excluded.



\## Reproducibility



By default, the random seed is auto-generated:



```python

RANDOM\_SEED\_BASE = None

```



For repeatable dataset generation, set a fixed integer value:



```python

RANDOM\_SEED\_BASE = 123456

```



\## Notes and Limitations



\- This project requires access to a local DIgSILENT PowerFactory installation.

\- The `powerfactory` Python module must be available through the configured PowerFactory Python path.

\- PowerFactory object names, study-case names, operation-scenario names, and grid names must match the active project.

\- Generated datasets may depend on the active PowerFactory model, switch-state CSV, and randomization settings.

\- The code is designed for controlled research and dataset-generation workflows, not for direct real-time protection operation.



