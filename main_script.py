# main_script.py
from __future__ import annotations

import sys
import logging
from datetime import datetime

from pf_adaptive_distance_dataset.core.config import Config, LOGS_DIR
from pf_adaptive_distance_dataset.core.models import DatasetMetadata, DatasetStatistics
from pf_adaptive_distance_dataset.pf_api.pf_session import PowerFactorySession
from pf_adaptive_distance_dataset.exports.export import stream_export_and_audit
from pf_adaptive_distance_dataset.pipeline.switch_states import load_switch_state_dataframe
from pf_adaptive_distance_dataset.pipeline.dataset_generator import generate_dataset_cases


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOGS_DIR / "pipeline.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


# ======================================================================================================================
# ------------------------------------------------ Pipeline Execution --------------------------------------------------
# ======================================================================================================================
def main():
    """
    Runs the full streamed distance-protection dataset generation pipeline.
    Workflow:
    - load switch-state configurations
    - open PowerFactory session
    - generate flat and graph-array payloads
    - stream exports and audit files
    - merge generation-side invalid-case reasons into export statistics
    """

    df_switch_states, l_switch_columns = load_switch_state_dataframe()

    i_total_switch_state_count = len(df_switch_states)

    if i_total_switch_state_count == 0:
        raise RuntimeError("No switch-state configurations were loaded.")

    logger.info(
        f"Loaded {i_total_switch_state_count} switch-state configurations."
    )

    logger.info("Opening PowerFactory session...")

    o_generation_statistics = DatasetStatistics()

    with PowerFactorySession(
        Config.PROJECT_NAME,
        Config.GRID_NAME,
    ) as o_session:
        logger.info(
            "Processing all switch states: "
            f"1 to {i_total_switch_state_count}..."
        )

        g_data_generator = generate_dataset_cases(
            session=o_session,
            stats=o_generation_statistics,
            df_chunk=df_switch_states,
            sw_cols=l_switch_columns,
            row_offset=0,
        )

        o_dataset_metadata = DatasetMetadata(
            dataset_version=Config.DATASET_VERSION,
            grid_name=Config.GRID_NAME,
            project_name=Config.PROJECT_NAME,
            generation_timestamp=datetime.now().isoformat(),
            random_seed_base=Config.get_random_seed_base(),
            line_randomization_enabled=Config.ENABLE_LINE_RANDOMIZATION,
            dg_randomization_enabled=Config.ENABLE_DG_CAPACITY_RANDOMIZATION,
            line_length_scale_range=(
                Config.LINE_LENGTH_SCALE_MIN,
                Config.LINE_LENGTH_SCALE_MAX,
            ),
            dg_capacity_scale_range=(
                Config.DG_CAPACITY_SCALE_MIN,
                Config.DG_CAPACITY_SCALE_MAX,
            ),
            notes=(
                "Single-run streamed grid protection dataset. "
                "Excel audit and final dataframe statistics are based on valid exported rows."
            ),
        )

        o_export_statistics = stream_export_and_audit(
            payloads=g_data_generator,
            metadata=o_dataset_metadata,
        )

    for s_invalid_reason, i_reason_count in (
        o_generation_statistics.invalid_case_reasons.items()
    ):
        o_export_statistics.invalid_case_reasons[s_invalid_reason] = (
            o_export_statistics.invalid_case_reasons.get(
                s_invalid_reason,
                0,
            )
            + i_reason_count
        )

    logger.info(
        "Dataset generation complete. "
        f"Valid rows: {o_export_statistics.total_cases_valid}/"
        f"{o_export_statistics.total_cases_generated}"
    )

    if o_export_statistics.invalid_case_reasons:
        logger.warning(
            "Invalid/generation issue summary: "
            f"{o_export_statistics.invalid_case_reasons}"
        )


if __name__ == "__main__":
    main()
