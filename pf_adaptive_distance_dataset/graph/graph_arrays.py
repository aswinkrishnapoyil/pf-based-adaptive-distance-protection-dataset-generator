# graph_arrays.py
from __future__ import annotations

import math
from typing import Any

from ..core.models import PhysicalEdge

from ..core.dataset_schema import (
    base_reach_columns,
    reach_infeed_correction_columns,
    target_reach_columns,
    directed_numeric_context_columns,
    directed_string_context_columns,
)

from .graph_array_utils import (
    clean_string,
    to_float,
    to_int,
    upper_triangle_index,
    flatten_row_major,
    canonical_corridor_id,
)


def collect_numeric_array(rows, col, default=0.0):
    """
    Collects one numeric column from all directed corridor rows.
    """

    l_rows = rows
    s_column_name = col
    f_default_value = default

    return [
        to_float(d_row.get(s_column_name), f_default_value)
        for d_row in l_rows
    ]


def collect_string_array(rows, col, default=""):
    """
    Collects one string column from all directed corridor rows.
    """

    l_rows = rows
    s_column_name = col
    s_default_value = default

    return [
        clean_string(d_row.get(s_column_name), s_default_value)
        for d_row in l_rows
    ]


def build_bus_index(rows):
    """
    Builds the graph bus index from relay and subsequent node IDs.
    Returns:
    - ordered bus ID list
    - bus ID to index dictionary
    - bus type list
    """

    l_rows = rows

    l_bus_ids = sorted(
        set(
            clean_string(d_row["relay_node_id"])
            for d_row in l_rows
        ).union(
            clean_string(d_row["subsequent_node_id"])
            for d_row in l_rows
        )
    )

    d_bus_index_by_id = {
        s_bus_id: i_bus_index
        for i_bus_index, s_bus_id in enumerate(l_bus_ids)
    }

    l_bus_type = [
        0
        for _s_bus_id in l_bus_ids
    ]

    return l_bus_ids, d_bus_index_by_id, l_bus_type


def extract_directed_edge_features(rows, bus_idx):
    """
    Extracts directed edge arrays from flat corridor rows.

    Also builds a physical undirected edge dictionary used later for Y-bus
    stamping.
    """

    l_rows = rows
    d_bus_index_by_id = bus_idx

    d_directed_edge_arrays: dict[str, list[Any]] = {
        s_key: []
        for s_key in [
            "directed_edge_relay_id",
            "directed_edge_from",
            "directed_edge_to",
            "directed_edge_from_index",
            "directed_edge_to_index",
            "directed_edge_id",
            "directed_edge_canonical_id",
            "directed_edge_line_is_in_service",
            "directed_edge_length_km",
            "directed_edge_r_ohm",
            "directed_edge_x_ohm",
            "directed_edge_hop_count",
            "directed_edge_is_parallel",
            "directed_edge_parallel_count",
            "directed_edge_parallel_id",
            "directed_edge_target_valid",
            "directed_edge_source_row_number",
        ]
    }

    d_physical_edges = {}

    for i_source_row_number, d_row in enumerate(l_rows, 1):
        s_source_bus_id = clean_string(d_row["relay_node_id"])
        s_target_bus_id = clean_string(d_row["subsequent_node_id"])

        i_source_bus_index = d_bus_index_by_id[s_source_bus_id]
        i_target_bus_index = d_bus_index_by_id[s_target_bus_id]

        s_corridor_id = clean_string(
            d_row["protected_corridor_id"]
        )

        s_canonical_corridor_id = canonical_corridor_id(
            s_corridor_id
        )

        f_corridor_length_km = to_float(
            d_row.get("protected_corridor_length_km"),
            0.0,
        )

        f_corridor_r_ohm = to_float(
            d_row.get("protected_corridor_r_ohm"),
            0.0,
        )

        f_corridor_x_ohm = to_float(
            d_row.get("protected_corridor_x_ohm"),
            0.0,
        )

        i_line_is_in_service = to_int(
            d_row.get("line_is_in_service"),
            1,
        )

        d_directed_edge_arrays["directed_edge_relay_id"].append(
            clean_string(d_row.get("relay_id"))
        )
        d_directed_edge_arrays["directed_edge_from"].append(
            s_source_bus_id
        )
        d_directed_edge_arrays["directed_edge_to"].append(
            s_target_bus_id
        )
        d_directed_edge_arrays["directed_edge_from_index"].append(
            i_source_bus_index
        )
        d_directed_edge_arrays["directed_edge_to_index"].append(
            i_target_bus_index
        )
        d_directed_edge_arrays["directed_edge_id"].append(
            s_corridor_id
        )
        d_directed_edge_arrays["directed_edge_canonical_id"].append(
            s_canonical_corridor_id
        )
        d_directed_edge_arrays["directed_edge_line_is_in_service"].append(
            i_line_is_in_service
        )
        d_directed_edge_arrays["directed_edge_length_km"].append(
            f_corridor_length_km
        )
        d_directed_edge_arrays["directed_edge_r_ohm"].append(
            f_corridor_r_ohm
        )
        d_directed_edge_arrays["directed_edge_x_ohm"].append(
            f_corridor_x_ohm
        )
        d_directed_edge_arrays["directed_edge_hop_count"].append(
            to_int(d_row.get("corridor_hop_count"), 1)
        )
        d_directed_edge_arrays["directed_edge_is_parallel"].append(
            to_int(d_row.get("protected_corridor_is_parallel"), 0)
        )
        d_directed_edge_arrays["directed_edge_parallel_count"].append(
            to_int(d_row.get("protected_corridor_parallel_count"), 1)
        )
        d_directed_edge_arrays["directed_edge_parallel_id"].append(
            clean_string(d_row.get("protected_corridor_parallel_id"))
        )
        d_directed_edge_arrays["directed_edge_target_valid"].append(1)
        d_directed_edge_arrays["directed_edge_source_row_number"].append(
            i_source_row_number
        )

        i_physical_from_index, i_physical_to_index = sorted(
            (
                i_source_bus_index,
                i_target_bus_index,
            )
        )

        t_physical_edge_key = (
            i_physical_from_index,
            i_physical_to_index,
            s_canonical_corridor_id,
        )

        if t_physical_edge_key not in d_physical_edges:
            d_physical_edges[t_physical_edge_key] = PhysicalEdge(
                i_physical_from_index,
                i_physical_to_index,
                s_canonical_corridor_id,
                s_corridor_id,
                f_corridor_length_km,
                f_corridor_r_ohm,
                f_corridor_x_ohm,
                i_line_is_in_service,
            )
        else:
            d_physical_edges[t_physical_edge_key].is_in_service = min(
                int(d_physical_edges[t_physical_edge_key].is_in_service),
                i_line_is_in_service,
            )

    d_directed_edge_metadata = {
        "directed_edge_count": len(
            d_directed_edge_arrays["directed_edge_id"]
        ),
        "directed_edge_target_valid_count": sum(
            d_directed_edge_arrays["directed_edge_target_valid"]
        ),
    }

    return (
        d_directed_edge_arrays,
        d_physical_edges,
        d_directed_edge_metadata,
    )


def stamp_switched_ybus(phys_edges, n):
    """
    Builds switched physical-edge arrays and Y-bus arrays from physical edges.

    Only in-service physical edges contribute to the Y-bus matrix.
    """

    d_physical_edges = phys_edges
    i_bus_count = n

    i_physical_pair_count = i_bus_count * (i_bus_count - 1) // 2

    l_lines_connected = [0] * i_physical_pair_count
    l_lines_connected_count = [0] * i_physical_pair_count
    l_y_lines = [complex(0, 0)] * i_physical_pair_count
    l_y_c_lines = [0.0] * i_physical_pair_count

    l_y_matrix = [
        [
            complex(0, 0)
            for _i_column_index in range(i_bus_count)
        ]
        for _i_row_index in range(i_bus_count)
    ]

    d_physical_edge_arrays = {
        s_key: []
        for s_key in [
            "physical_edge_from_index",
            "physical_edge_to_index",
            "physical_edge_id",
            "physical_edge_length_km",
            "physical_edge_r_ohm",
            "physical_edge_x_ohm",
            "physical_edge_y_real",
            "physical_edge_y_imag",
            "physical_edge_is_in_service",
        ]
    }

    for d_physical_edge in d_physical_edges.values():
        if not d_physical_edge.is_in_service:
            continue

        i_from_bus_index = d_physical_edge.a
        i_to_bus_index = d_physical_edge.b

        f_edge_r_ohm = float(d_physical_edge.r_ohm)
        f_edge_x_ohm = float(d_physical_edge.x_ohm)

        c_edge_impedance_ohm = complex(
            f_edge_r_ohm,
            f_edge_x_ohm,
        )

        c_edge_admittance = (
            complex(0, 0)
            if abs(c_edge_impedance_ohm) <= 1e-12
            else 1.0 / c_edge_impedance_ohm
        )

        i_upper_triangle_index = upper_triangle_index(
            i_from_bus_index,
            i_to_bus_index,
            i_bus_count,
        )

        l_lines_connected[i_upper_triangle_index] = 1
        l_lines_connected_count[i_upper_triangle_index] += 1
        l_y_lines[i_upper_triangle_index] += c_edge_admittance

        l_y_matrix[i_from_bus_index][i_from_bus_index] += c_edge_admittance
        l_y_matrix[i_to_bus_index][i_to_bus_index] += c_edge_admittance
        l_y_matrix[i_from_bus_index][i_to_bus_index] -= c_edge_admittance
        l_y_matrix[i_to_bus_index][i_from_bus_index] -= c_edge_admittance

        d_physical_edge_arrays["physical_edge_from_index"].append(
            i_from_bus_index
        )
        d_physical_edge_arrays["physical_edge_to_index"].append(
            i_to_bus_index
        )
        d_physical_edge_arrays["physical_edge_id"].append(
            clean_string(d_physical_edge.canonical_id)
        )
        d_physical_edge_arrays["physical_edge_length_km"].append(
            float(d_physical_edge.length_km)
        )
        d_physical_edge_arrays["physical_edge_r_ohm"].append(
            f_edge_r_ohm
        )
        d_physical_edge_arrays["physical_edge_x_ohm"].append(
            f_edge_x_ohm
        )
        d_physical_edge_arrays["physical_edge_y_real"].append(
            float(c_edge_admittance.real)
        )
        d_physical_edge_arrays["physical_edge_y_imag"].append(
            float(c_edge_admittance.imag)
        )
        d_physical_edge_arrays["physical_edge_is_in_service"].append(1)

    return {
        "Y_C_Lines": l_y_c_lines,
        "Lines_connected": l_lines_connected,
        "Lines_connected_count": l_lines_connected_count,
        "Y_Lines_real": [
            float(c_y_value.real)
            for c_y_value in l_y_lines
        ],
        "Y_Lines_imag": [
            float(c_y_value.imag)
            for c_y_value in l_y_lines
        ],
        "Y_matrix_real": [
            float(c_y_value.real)
            for c_y_value in flatten_row_major(l_y_matrix)
        ],
        "Y_matrix_imag": [
            float(c_y_value.imag)
            for c_y_value in flatten_row_major(l_y_matrix)
        ],
        "Y_matrix_real_2d": [
            [
                float(c_y_value.real)
                for c_y_value in l_matrix_row
            ]
            for l_matrix_row in l_y_matrix
        ],
        "Y_matrix_imag_2d": [
            [
                float(c_y_value.imag)
                for c_y_value in l_matrix_row
            ]
            for l_matrix_row in l_y_matrix
        ],
        "physical_edge_candidate_count": len(d_physical_edges),
        "physical_edge_count": len(
            d_physical_edge_arrays["physical_edge_id"]
        ),
        **d_physical_edge_arrays,
    }


def convert_scenario_to_graph_row(rows, meta, switch_status):
    """
    Converts all flat rows from one scenario into one graph-array row.
    """

    l_rows = rows
    d_metadata = meta
    l_switch_status = switch_status

    (
        l_bus_ids,
        d_bus_index_by_id,
        l_bus_type,
    ) = build_bus_index(l_rows)

    (
        d_directed_edge_data,
        d_physical_edges,
        d_directed_edge_metadata,
    ) = extract_directed_edge_features(
        l_rows,
        d_bus_index_by_id,
    )

    d_ybus_data = stamp_switched_ybus(
        d_physical_edges,
        len(l_bus_ids),
    )

    d_ml_array_data = {}

    d_ml_array_data.update(
        {
            s_column: collect_numeric_array(
                l_rows,
                s_column,
                0.0,
            )
            for s_column in directed_numeric_context_columns
        }
    )

    d_ml_array_data.update(
        {
            s_column: collect_string_array(
                l_rows,
                s_column,
                "",
            )
            for s_column in directed_string_context_columns
        }
    )

    d_ml_array_data.update(
        {
            s_column: collect_numeric_array(
                l_rows,
                s_column,
                math.nan,
            )
            for s_column in base_reach_columns
        }
    )

    d_ml_array_data.update(
        {
            s_column: collect_numeric_array(
                l_rows,
                s_column,
                math.nan,
            )
            for s_column in reach_infeed_correction_columns
        }
    )

    d_ml_array_data.update(
        {
            s_column: collect_numeric_array(
                l_rows,
                s_column,
                math.nan,
            )
            for s_column in target_reach_columns
        }
    )

    return {
        **d_metadata,
        **d_directed_edge_metadata,

        "switch_count": len(l_switch_status),
        "switch_status": l_switch_status,
        "switch_open_count_from_vector": (
            len(l_switch_status) - sum(l_switch_status)
        ),
        "switch_closed_count_from_vector": sum(l_switch_status),

        "bus_number": len(l_bus_ids),
        "bus_id": l_bus_ids,
        "bus_typ": l_bus_type,

        **d_ybus_data,
        **d_directed_edge_data,
        **d_ml_array_data,
    }
