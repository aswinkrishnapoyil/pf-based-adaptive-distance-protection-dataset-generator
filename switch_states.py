# switch_states.py
from __future__ import annotations

import os

import pandas as pd

from config import Config, PFAttr, SWITCH_STATE_FILE
from pf_utils import (
    get_pf_attribute,
    safe_set_attribute,
    get_unique_objects,
    is_object_inside_grid,
)


def _get_terminal_from_cubicle(cubicle):
    """
    Returns the terminal connected to a cubicle.
    """

    o_cubicle = cubicle

    return get_pf_attribute(o_cubicle, PFAttr.CTERM)


def _get_cubicle_switch_closed_state(cubicle):
    """
    Returns the first cubicle switch state as an integer flag.

    Returns:
        1 if the cubicle switch is closed or no switch is found.
        0 if the cubicle is missing or the switch is open.
    """

    o_cubicle = cubicle

    if o_cubicle is None:
        return 0

    try:
        l_switches = o_cubicle.GetChildren(1, "*.StaSwitch") or []

        if l_switches:
            i_switch_state = get_pf_attribute(
                l_switches[0],
                PFAttr.SWITCH_STATE,
                1,
                int,
            )

            return 1 if i_switch_state == 1 else 0

    except Exception:
        pass

    return 1


def load_switch_state_dataframe():
    """
    Loads switch-state scenarios from the configured switch-state file.

    Returns:
        tuple:
            - switch-state DataFrame
            - list of switch column names
    """

    if not Config.ENABLE_SWITCH_STATE_SCENARIOS:
        return pd.DataFrame([{"ConfigID": "live_grid_state"}]), []

    if not os.path.exists(SWITCH_STATE_FILE):
        raise RuntimeError(f"Switch file missing: {SWITCH_STATE_FILE}")

    df_switch_states = pd.read_csv(
        SWITCH_STATE_FILE,
        sep=None,
        engine="python",
    )

    l_switch_columns = [
        s_column
        for s_column in df_switch_states.columns
        if str(s_column).startswith("switch_")
    ]

    for s_switch_column in l_switch_columns:
        df_switch_states[s_switch_column] = (
            pd.to_numeric(
                df_switch_states[s_switch_column],
                errors="coerce",
            )
            .fillna(1)
            .apply(lambda value: 1 if int(value) == 1 else 0)
        )

    if Config.MAX_SWITCH_STATE_CONFIG_COUNT:
        df_switch_states = df_switch_states.head(
            Config.MAX_SWITCH_STATE_CONFIG_COUNT
        )

    return df_switch_states, l_switch_columns


def build_cubicle_lookup(app):
    """
    Builds a lookup dictionary from CIM RDF ID to PowerFactory cubicle object.
    """

    o_app = app

    d_cubicle_lookup = {}

    l_cubicles = o_app.GetCalcRelevantObjects("*.StaCubic") or []

    for o_cubicle in l_cubicles:
        rdf_id = get_pf_attribute(o_cubicle, "cimRdfId")

        if not rdf_id:
            continue

        s_rdf_value = (
            rdf_id[0]
            if isinstance(rdf_id, list)
            else str(rdf_id)
        )

        s_clean_rdf_value = s_rdf_value.lstrip("_").strip()

        if s_clean_rdf_value:
            d_cubicle_lookup[s_clean_rdf_value] = o_cubicle

    return d_cubicle_lookup


def get_first_cubicle_switch(cubicle):
    """
    Returns the first switch object connected to a cubicle.
    """

    o_cubicle = cubicle

    l_switches = (
        o_cubicle.GetChildren(1, "*.StaSwitch")
        if o_cubicle
        else []
    )

    return l_switches[0] if l_switches else None


def apply_switch_state(row, lookup, cfg):
    """
    Applies one switch-state row to the PowerFactory cubicle switches.
    """

    d_switch_state_row = row
    d_cubicle_lookup = lookup
    s_switch_state_config_id = cfg

    for s_column_name, switch_value in d_switch_state_row.items():
        if not str(s_column_name).startswith("switch_"):
            continue

        s_rdf_id = str(s_column_name).split("_", 1)[1].lstrip("_").strip()

        o_switch = get_first_cubicle_switch(
            d_cubicle_lookup.get(s_rdf_id)
        )

        if o_switch:
            safe_set_attribute(
                o_switch,
                PFAttr.SWITCH_STATE,
                1 if int(switch_value) == 1 else 0,
            )


def get_switch_state_outserv_controlled_objects(grid, app):
    """
    Collects grid objects whose outserv state can be controlled by switch states.
    """

    o_grid = grid
    o_app = app

    l_outserv_controlled_objects = []

    l_class_filters = [
        "*.ElmLne",
        "*.ElmGenstat",
        "*.ElmPvsys",
        "*.ElmLod",
        "*.ElmLodlv",
        "*.ElmTr2",
        "*.ElmTr3",
    ]

    for s_class_filter in l_class_filters:
        for o_object in o_app.GetCalcRelevantObjects(s_class_filter) or []:
            l_cubicles = [
                get_pf_attribute(o_object, s_attribute_name)
                for s_attribute_name in PFAttr.CUBICLE_ATTR_CANDIDATES
            ]

            l_valid_cubicles = [
                o_cubicle
                for o_cubicle in l_cubicles
                if o_cubicle is not None
            ]

            for o_cubicle in l_valid_cubicles:
                o_terminal = _get_terminal_from_cubicle(o_cubicle)

                if is_object_inside_grid(o_terminal, o_grid):
                    l_outserv_controlled_objects.append(o_object)
                    break

    return get_unique_objects(l_outserv_controlled_objects)


def apply_outserv_for_components_behind_open_switches(objs, orig_states):
    """
    Sets objects out of service if any connected cubicle switch is open.

    Objects that were originally out of service are left unchanged.
    """

    l_outserv_controlled_objects = objs
    d_original_outserv_states = orig_states

    for o_object in l_outserv_controlled_objects:
        if d_original_outserv_states.get(o_object, 0) != 0:
            continue

        l_cubicles = []

        for s_attribute_name in PFAttr.CUBICLE_ATTR_CANDIDATES:
            o_cubicle = get_pf_attribute(o_object, s_attribute_name)

            if o_cubicle is not None:
                l_cubicles.append(o_cubicle)

        if any(
            _get_cubicle_switch_closed_state(o_cubicle) == 0
            for o_cubicle in l_cubicles
        ):
            safe_set_attribute(
                o_object,
                PFAttr.OUTSERV,
                1,
            )
