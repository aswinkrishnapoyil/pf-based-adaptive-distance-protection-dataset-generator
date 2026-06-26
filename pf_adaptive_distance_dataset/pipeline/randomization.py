# randomization.py
from __future__ import annotations

import logging
import random
from typing import Optional

from ..core.config import Config, PFAttr

from ..pf_api.pf_utils import (
    get_safe_name,
    get_safe_full_name,
    get_pf_attribute,
    safe_set_attribute,
)

logger = logging.getLogger(__name__)


def restore_original_line_states(orig_lines: dict) -> None:
    """
    Restores line length and impedance values captured before randomization.
    Expected orig_lines format:
        {
            line_id: {
                "obj": line_object,
                "l": original_length_km,
                "r": original_r_ohm,
                "x": original_x_ohm,
            }
        }
    """

    d_original_line_states = orig_lines

    for _s_line_key, d_line_state in d_original_line_states.items():
        o_line = d_line_state.get("obj")

        if o_line is None:
            continue

        f_original_length_km = d_line_state.get("l", 0.0)
        f_original_r_ohm = d_line_state.get("r", 0.0)
        f_original_x_ohm = d_line_state.get("x", 0.0)

        safe_set_attribute(
            o_line,
            PFAttr.LINE_LENGTH,
            f_original_length_km,
        )

        safe_set_attribute(
            o_line,
            PFAttr.LINE_R,
            f_original_r_ohm,
        )

        safe_set_attribute(
            o_line,
            PFAttr.LINE_X,
            f_original_x_ohm,
        )


def restore_original_dg_capacity_states(orig_dgs: dict) -> None:
    """
    Restores DG capacity values captured before randomization.
    Expected orig_dgs format:
        {
            dg_id: {
                "obj": dg_object,
                "attr": capacity_attribute_name,
                "cap": original_capacity,
            }
        }
    """

    d_original_dg_states = orig_dgs

    for _s_dg_key, d_dg_state in d_original_dg_states.items():
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


def restore_original_randomization_states(
    orig_lines: dict,
    orig_dgs: dict,
) -> None:
    """
    Convenience restore for both line and DG randomization states.
    """

    d_original_line_states = orig_lines
    d_original_dg_states = orig_dgs

    restore_original_line_states(d_original_line_states)
    restore_original_dg_capacity_states(d_original_dg_states)


def apply_random_line_length_scenario(
    orig_lines: dict,
    scenario_id: str,
    seed: int,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
) -> list[dict]:
    """
    Applies random line scaling to length, R, and X.
    This keeps the line impedance per km approximately consistent by scaling:
        dline, R1, X1
    using the same random factor.
    """

    d_original_line_states = orig_lines
    s_scenario_id = scenario_id
    i_random_seed = seed

    f_scale_min = scale_min
    f_scale_max = scale_max

    if f_scale_min is None:
        f_scale_min = Config.LINE_LENGTH_SCALE_MIN

    if f_scale_max is None:
        f_scale_max = Config.LINE_LENGTH_SCALE_MAX

    o_random_generator = random.Random(int(i_random_seed))

    l_randomization_logs = []

    for s_line_key, d_line_state in d_original_line_states.items():
        o_line = d_line_state.get("obj")

        if o_line is None:
            continue

        f_original_length_km = float(
            d_line_state.get(
                "l",
                get_pf_attribute(
                    o_line,
                    PFAttr.LINE_LENGTH,
                    0.0,
                    float,
                ),
            )
            or 0.0
        )

        f_original_r_ohm = float(
            d_line_state.get(
                "r",
                get_pf_attribute(
                    o_line,
                    PFAttr.LINE_R,
                    0.0,
                    float,
                ),
            )
            or 0.0
        )

        f_original_x_ohm = float(
            d_line_state.get(
                "x",
                get_pf_attribute(
                    o_line,
                    PFAttr.LINE_X,
                    0.0,
                    float,
                ),
            )
            or 0.0
        )

        if f_original_length_km <= 0.0:
            continue

        f_scale_factor = o_random_generator.uniform(
            float(f_scale_min),
            float(f_scale_max),
        )

        f_randomized_length_km = f_original_length_km * f_scale_factor
        f_randomized_r_ohm = f_original_r_ohm * f_scale_factor
        f_randomized_x_ohm = f_original_x_ohm * f_scale_factor

        safe_set_attribute(
            o_line,
            PFAttr.LINE_LENGTH,
            f_randomized_length_km,
        )

        safe_set_attribute(
            o_line,
            PFAttr.LINE_R,
            f_randomized_r_ohm,
        )

        safe_set_attribute(
            o_line,
            PFAttr.LINE_X,
            f_randomized_x_ohm,
        )

        l_randomization_logs.append(
            {
                "scenario_id": s_scenario_id,
                "seed": i_random_seed,
                "object_id": s_line_key,
                "object_name": get_safe_name(o_line),
                "object_full_name": get_safe_full_name(o_line),
                "scale_factor": round(f_scale_factor, 6),
                "original_length_km": round(f_original_length_km, 6),
                "randomized_length_km": round(f_randomized_length_km, 6),
                "original_r_ohm": round(f_original_r_ohm, 6),
                "randomized_r_ohm": round(f_randomized_r_ohm, 6),
                "original_x_ohm": round(f_original_x_ohm, 6),
                "randomized_x_ohm": round(f_randomized_x_ohm, 6),
            }
        )

    logger.info(
        f"Applied line randomization for {s_scenario_id}: "
        f"{len(l_randomization_logs)} lines, "
        f"seed={i_random_seed}, "
        f"range=({f_scale_min}, {f_scale_max})"
    )

    return l_randomization_logs


def apply_random_dg_capacity_scenario(
    orig_dgs: dict,
    scenario_id: str,
    seed: int,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
) -> list[dict]:
    """Applies random DG capacity scaling."""

    d_original_dg_states = orig_dgs
    s_scenario_id = scenario_id
    i_random_seed = seed

    f_scale_min = scale_min
    f_scale_max = scale_max

    if f_scale_min is None:
        f_scale_min = Config.DG_CAPACITY_SCALE_MIN

    if f_scale_max is None:
        f_scale_max = Config.DG_CAPACITY_SCALE_MAX

    o_random_generator = random.Random(int(i_random_seed))

    l_randomization_logs = []

    for s_dg_key, d_dg_state in d_original_dg_states.items():
        o_dg = d_dg_state.get("obj")
        s_capacity_attribute_name = d_dg_state.get("attr")
        f_original_capacity_mva = d_dg_state.get("cap")

        if (
            o_dg is None
            or s_capacity_attribute_name is None
            or f_original_capacity_mva is None
        ):
            continue

        f_original_capacity_mva = float(f_original_capacity_mva)

        f_scale_factor = o_random_generator.uniform(
            float(f_scale_min),
            float(f_scale_max),
        )

        f_randomized_capacity_mva = (
            f_original_capacity_mva
            * f_scale_factor
        )

        safe_set_attribute(
            o_dg,
            s_capacity_attribute_name,
            f_randomized_capacity_mva,
        )

        l_randomization_logs.append(
            {
                "scenario_id": s_scenario_id,
                "seed": i_random_seed,
                "object_id": s_dg_key,
                "object_name": get_safe_name(o_dg),
                "object_full_name": get_safe_full_name(o_dg),
                "capacity_attribute": s_capacity_attribute_name,
                "scale_factor": round(f_scale_factor, 6),
                "original_capacity_mva": round(f_original_capacity_mva, 6),
                "randomized_capacity_mva": round(
                    f_randomized_capacity_mva,
                    6,
                ),
            }
        )

    logger.info(
        f"Applied DG capacity randomization for {s_scenario_id}: "
        f"{len(l_randomization_logs)} DGs, "
        f"seed={i_random_seed}, "
        f"range=({f_scale_min}, {f_scale_max})"
    )

    return l_randomization_logs
