# config.py
from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SCRIPT_DIR

RESULTS_DIR = PROJECT_ROOT / "Results"
OUTPUT_DIR = RESULTS_DIR
SWITCH_STATE_DIR = SCRIPT_DIR / "switch_state"
SWITCH_STATE_FILE = SWITCH_STATE_DIR / "switch_states.csv"
LOGS_DIR = SCRIPT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SWITCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    # UPDATE THESE TO MATCH YOUR LOCAL SETUP:
    PF_PYTHON_PATH = r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP3\Python\3.9"
    PROJECT_NAME = r"\ic84yhos\Venus\VeN2uS_ExampleGrid_v1.4_FRT_KP"
    GRID_NAME = "Grid_110kV.ElmNet"

    MASTER_STUDY_CASE_NAME = "SC_Master"
    MASTER_OPERATION_SCENARIO_NAME = "OS_Master"

    SLAVE_STUDY_CASE_PREFIX = "SC_Slave"
    SLAVE_OPERATION_SCENARIO_PREFIX = "OS_Slave"

    DATASET_VERSION = "v2.4.2"
    DATASET_EXPORT_TYPE = "streaming_graph_array_with_metadata"

    # --- Grading Factor for Zone Reach ---
    REACH_GF = 0.85
    ZONE3_REACH_FACTOR = 1.20
    ZERO_TOLERANCE = 1e-12

    # --- Switch State Controls ---
    ENABLE_SWITCH_STATE_SCENARIOS = True
    MAX_SWITCH_STATE_CONFIG_COUNT = None

    # --- Base Case & Randomization Volume ---
    INCLUDE_ORIGINAL_BASE_CASE = True
    RANDOMIZED_SCENARIO_COUNT = 25

    # --- Line Parameter Randomization ---
    ENABLE_LINE_RANDOMIZATION = True
    LINE_LENGTH_SCALE_MIN = 0.8
    LINE_LENGTH_SCALE_MAX = 1.2

    # --- Distributed Generation (DG) Randomization ---
    ENABLE_DG_CAPACITY_RANDOMIZATION = True
    DG_CAPACITY_SCALE_MIN = 0.8
    DG_CAPACITY_SCALE_MAX = 1.2

    RANDOM_SEED_BASE: Optional[int] = 20260707  # Set as None to get auto generated seed base
    DG_CAPACITY_RANDOM_SEED_OFFSET = 100000

    @classmethod
    def get_random_seed_base(cls) -> int:
        """Auto-generates a reproducible unique seed if one is not hardcoded."""
        if cls.RANDOM_SEED_BASE is None:
            timestamp = int(datetime.now().timestamp() * 1000) % 1000000
            seed = timestamp + (os.getpid() % 100000)
            cls.RANDOM_SEED_BASE = seed
            logging.getLogger(__name__).info(
                f"Auto-generated RANDOM_SEED_BASE: {cls.RANDOM_SEED_BASE}"
            )
        else:
            logging.getLogger(__name__).info(
                f"Using hardcoded RANDOM_SEED_BASE: {cls.RANDOM_SEED_BASE}"
            )
        return cls.RANDOM_SEED_BASE


class PFAttr:
    BUS1 = "bus1"
    BUS2 = "bus2"
    BUS3 = "bus3"

    BUSHV = "bushv"
    BUSMV = "busmv"
    BUSLV = "buslv"

    CTERM = "cterm"
    OUTSERV = "outserv"

    LINE_R = "R1"
    LINE_X = "X1"
    LINE_LENGTH = "dline"

    LINE_TYPE = "typ_id"

    SWITCH_STATE = "on_off"
    TERMINAL_USAGE = "iUsage"

    IKSS = "m:Ikss"
    IKSS_BUS1 = "m:Ikss:bus1"
    IKSS_BUS2 = "m:Ikss:bus2"

    DG_CAPACITY_CANDIDATES = ["sgn", "Sn", "snom", "Srated"]

    CUBICLE_ATTR_CANDIDATES = [
        BUS1,
        BUS2,
        BUS3,
        BUSHV,
        BUSMV,
        BUSLV,
    ]

    IKSS_CANDIDATES = [
        IKSS,
        IKSS_BUS1,
        IKSS_BUS2,
    ]