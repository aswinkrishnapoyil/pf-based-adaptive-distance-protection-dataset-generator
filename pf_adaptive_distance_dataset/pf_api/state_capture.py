# state_capture.py
from __future__ import annotations

from config import PFAttr

from pf_utils import (
    get_safe_full_name,
    get_pf_attribute,
    get_unique_objects,
    is_object_inside_grid,
)

from topology import (
    get_terminal_from_cubicle,
    get_line_terminals,
    get_line_impedance,
    get_line_length_value,
)

from switch_states import (
    build_cubicle_lookup,
    get_first_cubicle_switch,
    get_switch_state_outserv_controlled_objects,
)


def capture_original_state_from_active_slave(grid, app):
    """
    Capture original line, DG, switch, and outserv states
    from the currently active slave study case/scenario.
    This must be called AFTER:
    - slave study case is created
    - slave operation scenario is created
    - both are activated
    """

    o_grid = grid
    o_app = app

    # ------------------------------------------------------------
    # 1. Capture all relevant lines in the target grid
    # ------------------------------------------------------------
    l_lines = []

    for o_line in o_app.GetCalcRelevantObjects("*.ElmLne") or []:
        try:
            (
                o_terminal_1,
                o_terminal_2,
                _o_cubicle_1,
                _o_cubicle_2,
            ) = get_line_terminals(o_line)

            if not (
                is_object_inside_grid(o_terminal_1, o_grid)
                or is_object_inside_grid(o_terminal_2, o_grid)
            ):
                continue

            f_line_length_km = get_line_length_value(o_line)

            if f_line_length_km <= 0:
                continue

            l_lines.append(o_line)

        except Exception:
            continue

    l_lines = get_unique_objects(l_lines)

    d_original_line_states = {}

    for o_line in l_lines:
        f_line_r_ohm, f_line_x_ohm = get_line_impedance(o_line)

        d_original_line_states[get_safe_full_name(o_line)] = {
            "obj": o_line,
            "l": get_line_length_value(o_line),
            "r": f_line_r_ohm,
            "x": f_line_x_ohm,
        }

    # ------------------------------------------------------------
    # 2. Capture all DG capacities in the target grid
    # ------------------------------------------------------------
    l_distributed_generators = []

    for s_object_filter in ["*.ElmGenstat", "*.ElmPvsys"]:
        for o_dg in o_app.GetCalcRelevantObjects(s_object_filter) or []:
            try:
                o_dg_cubicle = get_pf_attribute(o_dg, PFAttr.BUS1)
                o_dg_terminal = get_terminal_from_cubicle(o_dg_cubicle)

                if is_object_inside_grid(o_dg_terminal, o_grid):
                    l_distributed_generators.append(o_dg)

            except Exception:
                continue

    l_distributed_generators = get_unique_objects(l_distributed_generators)

    d_original_dg_states = {}

    for o_dg in l_distributed_generators:
        for s_capacity_attribute_name in PFAttr.DG_CAPACITY_CANDIDATES:
            f_capacity_mva = get_pf_attribute(
                o_dg,
                s_capacity_attribute_name,
                None,
                float,
            )

            if f_capacity_mva is not None:
                d_original_dg_states[get_safe_full_name(o_dg)] = {
                    "obj": o_dg,
                    "attr": s_capacity_attribute_name,
                    "cap": f_capacity_mva,
                }
                break

    # ------------------------------------------------------------
    # 3. Capture switch states
    # ------------------------------------------------------------
    d_cubicle_lookup = build_cubicle_lookup(o_app)

    d_original_switch_states = {}

    for s_rdf_identifier, o_cubicle in d_cubicle_lookup.items():
        o_switch = get_first_cubicle_switch(o_cubicle)

        if o_switch is not None:
            i_original_switch_state = int(
                get_pf_attribute(
                    o_switch,
                    PFAttr.SWITCH_STATE,
                    1,
                    int,
                )
            )

            d_original_switch_states[s_rdf_identifier] = {
                "sw": o_switch,
                "v": i_original_switch_state,
            }

    # ------------------------------------------------------------
    # 4. Capture outserv states
    # ------------------------------------------------------------
    l_outserv_controlled_objects = get_switch_state_outserv_controlled_objects(
        o_grid,
        o_app,
    )

    d_original_outserv_states = {
        o_object: get_pf_attribute(
            o_object,
            PFAttr.OUTSERV,
            0,
            int,
        )
        for o_object in l_outserv_controlled_objects
    }

    l_cached_all_grid_dg = l_distributed_generators

    return (
        d_cubicle_lookup,
        d_original_line_states,
        d_original_dg_states,
        d_original_outserv_states,
        d_original_switch_states,
        l_outserv_controlled_objects,
        l_cached_all_grid_dg,
    )
