# grid_state.py
from __future__ import annotations

from ..core.config import PFAttr
from .pf_utils import safe_set_attribute


def restore_grid_state(orig_lines, orig_dgs, orig_out, orig_sw):
    """
    Restore the active slave grid state to the captured baseline.
    Restores:
    - original line type assignment
    - original line length
    - DG capacity
    - component outserv states
    - switch on/off states

    Line R1/X1 are not restored directly. In the approved production
    approach, line randomization is performed only through dline.
    """

    d_original_line_states = orig_lines
    d_original_dg_states = orig_dgs
    d_original_outserv_states = orig_out
    d_original_switch_states = orig_sw

    # ------------------------------------------------------------------
    # Restore original line type and line length
    # ------------------------------------------------------------------
    for d_line_state in d_original_line_states.values():
        o_line = d_line_state.get("obj")

        if o_line is None:
            continue

        f_original_line_length_km = d_line_state.get("l", 0.0)
        o_original_line_type = d_line_state.get("typ_id")

        # First restore the original line type, so the line no longer points
        # to the temporary cloned type.
        if o_original_line_type is not None:
            safe_set_attribute(
                o_line,
                PFAttr.LINE_TYPE,
                o_original_line_type,
            )

        # Restore original line length.
        safe_set_attribute(
            o_line,
            PFAttr.LINE_LENGTH,
            f_original_line_length_km,
        )

    # ------------------------------------------------------------------
    # Restore distributed generator capacity values
    # ------------------------------------------------------------------
    for d_dg_state in d_original_dg_states.values():
        o_dg = d_dg_state.get("obj")
        s_capacity_attribute_name = d_dg_state.get("attr")
        f_original_capacity_mva = d_dg_state.get("cap")

        if (
            o_dg is None
            or s_capacity_attribute_name is None
            or f_original_capacity_mva is None
        ):
            continue

        safe_set_attribute(
            o_dg,
            s_capacity_attribute_name,
            f_original_capacity_mva,
        )

    # ------------------------------------------------------------------
    # Restore original outserv states
    # ------------------------------------------------------------------
    for o_object, i_original_outserv in d_original_outserv_states.items():
        safe_set_attribute(
            o_object,
            PFAttr.OUTSERV,
            i_original_outserv,
        )

    # ------------------------------------------------------------------
    # Restore original switch on/off states
    # ------------------------------------------------------------------
    for d_switch_state in d_original_switch_states.values():
        o_switch = d_switch_state.get("sw")
        i_original_switch_state = d_switch_state.get("v")

        if o_switch is None or i_original_switch_state is None:
            continue

        safe_set_attribute(
            o_switch,
            PFAttr.SWITCH_STATE,
            i_original_switch_state,
        )
