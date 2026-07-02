# view_parquet.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

# Select which full grid/scenario row to display.
# Row 0 = first full grid scenario in the ML-ready graph-array parquet file.
CASE_ROW_INDEX = 5

# Keep this as None to automatically select the newest ML-ready parquet file.
PARQUET_FILE_PATH = None

# Example for fixed file:
# PARQUET_FILE_PATH = r"Results\distance_protection_graph_array_ml_ready_20260626_194904.parquet"

# Save the displayed row to a text file.
SAVE_TO_TXT = False


# =============================================================================
# FUNCTIONS
# =============================================================================

def find_latest_ml_ready_graph_parquet(results_dir: Path) -> Path:
    """
    Finds the newest ML-ready graph-array parquet file in Results/.
    """

    l_ml_ready_parquet_files = sorted(
        results_dir.glob("distance_protection_graph_array_ml_ready_*.parquet"),
        key=lambda p_file: p_file.stat().st_mtime,
        reverse=True,
    )

    if not l_ml_ready_parquet_files:
        raise FileNotFoundError(
            "No ML-ready graph-array parquet files found in Results folder: "
            f"{results_dir}\n\n"
            "Expected filename pattern:\n"
            "distance_protection_graph_array_ml_ready_*.parquet"
        )

    return l_ml_ready_parquet_files[0]


def view_full_ml_ready_grid_case_as_pandas_row(
    file_path: str | Path,
    row_index: int,
) -> str:
    """
    Prints one full ML-ready grid/scenario case in pandas Series.to_string() format.
    """

    p_file_path = Path(file_path)

    if not p_file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {p_file_path}")

    df_data = pd.read_parquet(
        p_file_path,
        engine="pyarrow",
    )

    if row_index < 0 or row_index >= len(df_data):
        raise IndexError(
            f"row_index={row_index} is outside valid range "
            f"0 to {len(df_data) - 1}"
        )

    sr_row = df_data.iloc[row_index]

    f_file_size_mb = os.path.getsize(p_file_path) / (1024 * 1024)

    s_header = (
        f"ML-ready Parquet file: {p_file_path}\n"
        f"File size: {f_file_size_mb:.2f} MB\n"
        f"Total full grid cases: {len(df_data)}\n"
        f"Columns per grid case: {len(df_data.columns)}\n"
        f"Displayed row index: {row_index}\n\n"
    )

    s_row_text = sr_row.to_string()

    s_output_text = s_header + s_row_text

    sys.stdout.buffer.write(
        s_output_text.encode("utf-8", errors="replace") + b"\n"
    )

    return s_output_text


def save_output_text(
    results_dir: Path,
    output_text: str,
    row_index: int,
) -> None:
    """
    Saves the pandas-style row output to a TXT file.
    """

    p_output_path = (
        results_dir / f"ml_ready_parquet_full_grid_case_row_{row_index:04d}.txt"
    )

    with p_output_path.open("w", encoding="utf-8") as o_file:
        o_file.write(output_text)

    print(f"\nSaved output to: {p_output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Make pandas display long lists/strings without truncating.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.width", 1000)

    # This script should be in the repository root.
    p_project_root = Path(__file__).resolve().parent
    p_results_dir = p_project_root / "Results"

    if PARQUET_FILE_PATH is None:
        p_parquet_file_path = find_latest_ml_ready_graph_parquet(p_results_dir)
    else:
        p_parquet_file_path = Path(PARQUET_FILE_PATH)

    s_output = view_full_ml_ready_grid_case_as_pandas_row(
        file_path=p_parquet_file_path,
        row_index=CASE_ROW_INDEX,
    )

    if SAVE_TO_TXT:
        save_output_text(
            results_dir=p_results_dir,
            output_text=s_output,
            row_index=CASE_ROW_INDEX,
        )
