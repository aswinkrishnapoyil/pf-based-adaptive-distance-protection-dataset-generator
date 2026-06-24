# =============================================================================
# v0.1.py
#
# Unified corridor-based feature matrix generation
# with switch-state topology scenarios, randomized grid-scenario export,
# and ML-ready dataset export
#
# Purpose:
#   - Load manually created switch-state topology scenarios from Switch_state.csv
#   - Apply each switch-state configuration to the PowerFactory grid
#   - Detect protected busbar-to-busbar corridors for each active topology
#   - Process each corridor one by one
#   - Build one complete final row per relay/corridor direction
#   - Keep Zone 2 / Zone 3 stabilized branch-selection logic
#   - Add zone-based turbine/infeed summary directly to each final row
#
# Scenario generation:
#   - Use switch states as the outer topology-scenario loop:
#         ss_0001, ss_0002, ...
#   - Include the original base case for each switch state as scenario base_0000
#   - Generate randomized scenarios for each switch state:
#         rand_0001, rand_0002, ...
#   - Randomize line length dline together with R1 and X1
#   - Randomize DG installed capacity using the detected DG capacity attribute
#   - Restore original line, DG, component outserv, and switch states before
#     every switch-state scenario
#   - Re-apply the active switch-state configuration before base_0000 and
#     before every randomized scenario
#   - Derive temporary component outserv states from open cubicle switches
#     before base_0000 and before every randomized scenario
#   - Restore original line, DG, component outserv, and switch states again
#     after all scenarios
#
# Switch-state topology handling:
#   - Switch_state.csv uses:
#         ConfigID
#         switch_<cubicle_cimRdfId>
#   - Switch values are interpreted as:
#         1 = closed
#         0 = open
#   - Each switch column is matched to a StaCubic.cimRdfId
#   - The corresponding child StaSwitch.on_off value is updated
#   - Switch states are applied directly to PowerFactory cubicle switches
#   - NetworkX/corridor topology extraction explicitly ignores
#     open-switched branches
#   - After each switch-state row is applied, components connected behind
#     open cubicle switches are temporarily set out of service:
#         ElmLne, ElmGenstat, ElmPvsys, ElmLod, ElmLodlv, ElmTr2, ElmTr3
#   - Original outserv states are restored before every switch state,
#     before every randomized scenario, and finally after the full run.
#
# Outputs:
#   - Full engineering feature matrix:
#       case_feature_matrix_randomized_grid_scenarios.csv/.xlsx
#   - ML-ready numeric feature matrix:
#       case_feature_matrix_randomized_grid_scenarios_ml_ready.csv/.xlsx
#   - Trace index for mapping ML rows back to engineering cases:
#       case_feature_matrix_randomized_grid_scenarios_trace_index.csv/.xlsx
#   - Line length/R/X randomization log:
#       line_length_randomization_log.csv/.xlsx
#   - DG capacity randomization log:
#       dg_capacity_randomization_log.csv/.xlsx
#   - Switch-state application log:
#       switch_state_application_log.csv/.xlsx
#   - Reach percentage diagnostics:
#       reach_percentage_diagnostics.csv/.xlsx
#   - Switch-state component outserv derivation log:
#       switch_state_component_outserv_log.csv/.xlsx
#
# Notes:
#   - The full engineering dataset keeps object IDs, names, methods,
#     scenario identifiers, switch-state identifiers, and diagnostic
#     traceability columns.
#   - The ML-ready dataset removes IDs, names, method strings, random seeds,
#     scenario identifiers, and intermediate correction labels.
#   - Numeric switch-state summary columns can remain as ML features:
#         switch_state_open_switch_count
#         switch_state_closed_switch_count
#   - Final target reach columns are kept as ML labels.
# =============================================================================


# =============================================================================
# Imports
# =============================================================================

import os
import sys
import random
from collections import defaultdict

import pandas as pd

# =============================================================================
# PowerFactory Python Path
# =============================================================================

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP3\Python\3.9")

import powerfactory

# =============================================================================
# User Settings
# =============================================================================

s_dir_codingbase = r"Z:\Studenten\KrishnaPoyil\Thesis\_codebase\distance-protection-parameter-prediction"
s_projectname = r"\ic84yhos\Venus\VeN2uS_ExampleGrid_v1.4_FRT_KP"
s_grid_name = "Grid_110kV.ElmNet"

s_result_dir = os.path.join(
    s_dir_codingbase,
    "Results",
    "Ybus Output",
)

b_enable_debug = True

# =============================================================================
# Global Settings
# =============================================================================

i_reach_gf = 0.85
f_zone3_reach_factor = 1.20
f_zero_tolerance = 1e-12
i_max_traversal_depth = 50

# =============================================================================
# Randomized Line-Length Scenario Settings
# =============================================================================

b_enable_line_length_randomization = True

i_randomized_scenario_count = 10

i_random_seed_base = 1000

f_line_length_scale_min = 0.8
f_line_length_scale_max = 1.2

b_include_original_base_case_in_randomized_export = True

# =============================================================================
# Randomized DG Capacity Scenario Settings
# =============================================================================

b_enable_dg_capacity_randomization = True

f_dg_capacity_scale_min = 0.8
f_dg_capacity_scale_max = 1.2

i_dg_capacity_random_seed_offset = 100000

# =============================================================================
# Switch-State Scenario Settings
# =============================================================================

b_enable_switch_state_scenarios = True

s_switch_state_dir = os.path.join(
    s_dir_codingbase,
    "Results",
    "Switch State",
)

s_switch_state_file = os.path.join(
    s_switch_state_dir,
    "Switch_state.csv",
)

# Use 2 for first test.
# Use None for full run with all manually created switch states.
i_max_switch_state_config_count = None

# If enabled, components connected behind open cubicle switches are
# temporarily set out of service for the active switch-state scenario.
b_enable_switch_state_outserv_derivation = True

# =============================================================================
# Global PowerFactory Handle
# =============================================================================

o_pf = None

# =============================================================================
# Output Columns
# =============================================================================

l_case_feature_columns = [
    # =========================================================================
    # PART 1: Protected corridor topology and electrical features
    # =========================================================================
    "case_id",
    "relay_id",
    "relay_node_id",
    "protected_corridor_id",
    "subsequent_node_id",
    "line_is_in_service",
    "corridor_hop_count",
    "protected_corridor_length_km",
    "protected_corridor_r_ohm",
    "protected_corridor_x_ohm",
    "protected_corridor_is_parallel",
    "protected_corridor_parallel_count",
    "protected_corridor_parallel_id",

    # =========================================================================
    # PART 2: Subsequent-node downstream context
    # =========================================================================
    "next_node_busbar_count",
    "next_node_junction_node_count",
    "next_node_distributed_generation_count",

    # =========================================================================
    # PART 3: Shortest downstream / Zone 2 branch context
    # =========================================================================
    "shortest_downstream_branch_id",
    "shortest_downstream_branch_remote_node_id",
    "shortest_downstream_branch_hop_count",
    "shortest_downstream_branch_length_km",
    "shortest_downstream_branch_r_ohm",
    "shortest_downstream_branch_x_ohm",
    "shortest_downstream_branch_has_parallel",
    "shortest_downstream_branch_parallel_count",
    "shortest_downstream_branch_parallel_id",
    "shortest_downstream_branch_junction_node_count",
    "shortest_downstream_branch_junction_node_id",
    "shortest_downstream_branch_distributed_generation_count",
    "shortest_downstream_branch_distributed_generation_id",

    # =========================================================================
    # PART 4: Zone 2 branch-selection logic
    # =========================================================================
    "zone2_branch_selection_method",
    "zone2_selected_branch_id",
    "zone2_reach_calculation_method",
    "zone2_impedance_basis",

    "parallel_group_count_forward",
    "parallel_group_id",
    "zone2_parallel_branch_for_complex_length_km",
    "zone2_parallel_branch_for_complex_r_ohm",
    "zone2_parallel_branch_for_complex_x_ohm",

    # =========================================================================
    # PART 5: Longest downstream / Zone 3 branch context
    # =========================================================================
    "zone3_branch_selection_method",
    "zone3_selected_branch_id",
    "longest_downstream_branch_hop_count",
    "longest_downstream_branch_length_km",
    "longest_downstream_branch_r_ohm",
    "longest_downstream_branch_x_ohm",

    # =========================================================================
    # PART 6: Distributed generation location features
    # =========================================================================
    "relay_busbar_distributed_generation_count",
    "relay_busbar_distributed_generation_id",
    "relay_busbar_distributed_generation_capacity_mva",

    "protected_corridor_distributed_generation_count",
    "protected_corridor_distributed_generation_id",
    "protected_corridor_distributed_generation_capacity_mva",

    "subsequent_busbar_distributed_generation_count",
    "subsequent_busbar_distributed_generation_id",
    "subsequent_busbar_distributed_generation_capacity_mva",

    "downstream_branch_distributed_generation_count",
    "downstream_branch_distributed_generation_id",
    "downstream_branch_distributed_generation_capacity_mva",

    "remote_busbar_distributed_generation_count",
    "remote_busbar_distributed_generation_id",
    "remote_busbar_distributed_generation_capacity_mva",

    # =========================================================================
    # PART 7: Zone 1 / Zone 2 turbine pool features
    # =========================================================================
    "zone1_turbines_candidate_count",
    "zone1_turbines_candidate_id",
    "zone1_turbines_considered_count",
    "zone1_turbines_considered_id",
    "zone1_turbines_skipped_count",
    "zone1_turbines_skipped_id",
    "zone1_turbines_total_capacity_mva",

    "zone2_turbines_candidate_count",
    "zone2_turbines_candidate_id",
    "zone2_turbines_considered_count",
    "zone2_turbines_considered_id",
    "zone2_turbines_skipped_count",
    "zone2_turbines_skipped_id",
    "zone2_turbines_total_capacity_mva",

    # =========================================================================
    # PART 8: Base reaches
    # =========================================================================
    "base_zone1_r_reach_ohm",
    "base_zone1_x_reach_ohm",

    "base_zone2_r_reach_ohm",
    "base_zone2_x_reach_ohm",

    "base_zone3_r_reach_ohm",
    "base_zone3_x_reach_ohm",

    # =========================================================================
    # PART 9: Ground-truth correction labels
    # =========================================================================
    "zone1_infeed_correction_r_ohm",
    "zone1_infeed_correction_x_ohm",

    "zone2_infeed_correction_r_ohm",
    "zone2_infeed_correction_x_ohm",

    # =========================================================================
    # PART 10: Final target labels
    # =========================================================================
    "target_zone1_r_reach_ohm",
    "target_zone1_x_reach_ohm",

    "target_zone2_r_reach_ohm",
    "target_zone2_x_reach_ohm",

    "target_zone3_r_reach_ohm",
    "target_zone3_x_reach_ohm",
]


# =============================================================================
# Print Utility
# =============================================================================

def _print(s_message):
    print(str(s_message), flush=True)

    try:
        if o_pf is not None:
            o_pf.PrintPlain(str(s_message))
    except Exception:
        pass


# =============================================================================
# PowerFactory Initialisation Functions
# =============================================================================

def connect_to_powerfactory(b_enable_debug=False):
    _print("\n🔌 Connecting to PowerFactory...")

    try:
        o_application = powerfactory.GetApplicationExt()
    except Exception:
        o_application = powerfactory.GetApplication()

    if o_application is None:
        raise RuntimeError(
            "❌ PowerFactory connection failed. Ensure PowerFactory is installed and licensed."
        )

    try:
        o_application.Show()
    except Exception as o_error:
        _print(f"⚠️ Could not show PowerFactory window: {o_error}")

    if b_enable_debug:
        try:
            o_application.EchoOn()
        except Exception:
            pass

    _print("✅ PowerFactory connected.")

    return o_application


def prepare_working_directory(s_directory):
    os.makedirs(s_directory, exist_ok=True)
    os.chdir(s_directory)

    _print(f"✅ Working directory set to: {s_directory}")


def activate_powerfactory_project(o_application, s_projectname):
    _print("\n⚙️ Activating PowerFactory project...")

    if o_application.ActivateProject(s_projectname) != 0:
        raise RuntimeError(f"❌ Could not activate project: {s_projectname}")

    o_project = o_application.GetActiveProject()

    if o_project is None:
        raise RuntimeError("❌ No active PowerFactory project found.")

    _print(f"✅ Project activated: {s_projectname}")

    return o_project


def get_target_grid(o_project, s_grid_name):
    _print("\n⚙️ Selecting target grid...")

    l_grid_objects = o_project.GetContents(s_grid_name, 1)

    if not l_grid_objects:
        raise RuntimeError(f"❌ Grid not found: {s_grid_name}")

    o_grid = l_grid_objects[0]

    _print(f"✅ Selected grid: {o_grid.loc_name}")

    return o_grid


# =============================================================================
# Safe PowerFactory Helper Functions
# =============================================================================

def _get_safe_name(o_object):
    try:
        if o_object is None:
            return "UNKNOWN"

        return str(o_object.loc_name)

    except Exception:
        return "UNKNOWN"


def _get_safe_class_name(o_object):
    try:
        if o_object is None:
            return "UNKNOWN"

        return str(o_object.GetClassName())

    except Exception:
        return "UNKNOWN"


def _get_safe_full_name(o_object):
    try:
        if o_object is None:
            return "UNKNOWN"

        return str(o_object.GetFullName())

    except Exception:
        return _get_safe_name(o_object)


def _is_object_in_service(o_object):
    try:
        return int(o_object.outserv) == 0

    except Exception:
        return True


def _is_object_inside_grid(o_object, o_grid):
    try:
        if o_object is None or o_grid is None:
            return False

        s_grid_full_name = o_grid.GetFullName()
        o_parent = o_object

        while o_parent is not None:
            try:
                if o_parent.GetFullName() == s_grid_full_name:
                    return True

                o_parent = o_parent.GetParent()

            except Exception:
                break

        return s_grid_full_name in o_object.GetFullName()

    except Exception:
        return False


def _get_attribute_object(o_object, s_attribute):
    if o_object is None:
        return None

    try:
        return o_object.GetAttribute(s_attribute)

    except Exception:
        try:
            return getattr(o_object, s_attribute)

        except Exception:
            return None


def _get_attribute_float(o_object, s_attribute, f_default=None):
    if o_object is None:
        return f_default

    try:
        f_value = o_object.GetAttribute(s_attribute)

        if f_value is None:
            return f_default

        return float(f_value)

    except Exception:
        try:
            f_value = getattr(o_object, s_attribute)

            if f_value is None:
                return f_default

            return float(f_value)

        except Exception:
            return f_default


def _get_attribute_int(o_object, s_attribute, i_default=None):
    if o_object is None:
        return i_default

    try:
        i_value = o_object.GetAttribute(s_attribute)

        if i_value is None:
            return i_default

        return int(i_value)

    except Exception:
        try:
            i_value = getattr(o_object, s_attribute)

            if i_value is None:
                return i_default

            return int(i_value)

        except Exception:
            return i_default


def _get_first_available_float(o_object, l_attributes):
    for s_attribute in l_attributes:
        f_value = _get_attribute_float(o_object, s_attribute, None)

        if f_value is not None:
            return f_value, s_attribute

    return None, None


def _get_terminal_from_cubicle(o_cubicle):
    if o_cubicle is None:
        return None

    try:
        return o_cubicle.GetAttribute("cterm")

    except Exception:
        try:
            return o_cubicle.cterm

        except Exception:
            return None


def _get_opposite_terminal(o_line, o_terminal):
    try:
        o_terminal_1, o_terminal_2, _, _ = get_line_terminals(o_line)

        if o_terminal_1 == o_terminal:
            return o_terminal_2

        if o_terminal_2 == o_terminal:
            return o_terminal_1

    except Exception:
        pass

    return None


def _get_line_impedance(o_line):
    f_line_r_ohm = _get_attribute_float(o_line, "R1", 0.0)
    f_line_x_ohm = _get_attribute_float(o_line, "X1", 0.0)

    return f_line_r_ohm, f_line_x_ohm


def _get_relay_side_and_cubicle(o_relay_line, o_start_terminal):
    try:
        o_cubicle_bus1 = _get_attribute_object(o_relay_line, "bus1")
        o_cubicle_bus2 = _get_attribute_object(o_relay_line, "bus2")

        o_terminal_bus1 = _get_terminal_from_cubicle(o_cubicle_bus1)
        o_terminal_bus2 = _get_terminal_from_cubicle(o_cubicle_bus2)

        if o_terminal_bus1 == o_start_terminal:
            return (
                o_cubicle_bus1,
                _get_safe_name(o_cubicle_bus1),
                "m:Ikss:bus1",
                "bus1",
            )

        if o_terminal_bus2 == o_start_terminal:
            return (
                o_cubicle_bus2,
                _get_safe_name(o_cubicle_bus2),
                "m:Ikss:bus2",
                "bus2",
            )

    except Exception:
        pass

    return None, "UNKNOWN", "m:Ikss:bus1", "unknown"


def _read_relay_reference_ikss(o_reference_cubicle, o_relay_line, s_relay_ikss_line_attr):
    f_ref_ikss = _get_attribute_float(
        o_relay_line,
        s_relay_ikss_line_attr,
        None,
    )

    if f_ref_ikss is not None and f_ref_ikss > 0.0:
        return f_ref_ikss, f"line.{s_relay_ikss_line_attr}"

    f_ref_ikss_gen = _get_attribute_float(
        o_relay_line,
        "m:Ikss",
        None,
    )

    if f_ref_ikss_gen is not None and f_ref_ikss_gen > 0.0:
        return f_ref_ikss_gen, "line.m:Ikss"

    if o_reference_cubicle is not None:
        f_ref_ikss_cub, s_attr = _get_first_available_float(
            o_reference_cubicle,
            ["m:Ikss", "m:Ikss:bus1", "m:Ikss:bus2"],
        )

        if f_ref_ikss_cub is not None and f_ref_ikss_cub > 0.0:
            return f_ref_ikss_cub, f"cubicle.{s_attr}"

    return 0.0, "NOT_FOUND_OR_ZERO"


def _safe_set_attribute(
        o_object,
        s_attribute,
        value,
):
    """
    Safely sets a PowerFactory attribute.
    """

    if o_object is None:
        return False

    try:
        o_object.SetAttribute(
            s_attribute,
            value,
        )
        return True

    except Exception:
        pass

    try:
        setattr(
            o_object,
            s_attribute,
            value,
        )
        return True

    except Exception:
        return False


def get_short_circuit_command(o_project):
    """
    Returns the active short-circuit command object.
    """

    try:
        o_shc = o_pf.GetFromStudyCase(
            "ComShc",
        )

        if o_shc is not None:
            return o_shc

    except Exception:
        pass

    try:
        l_shc_objects = o_project.GetContents(
            "Short-Circuit Calculation*",
            1,
        )

        if l_shc_objects:
            return l_shc_objects[0]

    except Exception:
        pass

    return None


def configure_short_circuit_command(o_shc):
    """
    Configures the short-circuit command for a 3-phase short circuit.
    """

    if o_shc is None:
        return

    # These are deliberately protected with safe setters because PowerFactory
    # attribute names can vary between versions/study-case templates.
    _safe_set_attribute(o_shc, "iopt_allbus", 0)
    _safe_set_attribute(o_shc, "iopt_mode", 3)
    _safe_set_attribute(o_shc, "iopt_mde", 3)
    _safe_set_attribute(o_shc, "ip_flt", 0)
    _safe_set_attribute(o_shc, "iopt_opt", "pro")
    _safe_set_attribute(o_shc, "iopt_cur", 0)


def get_ikss_value(o_object):
    """
    Reads Ikss from a PowerFactory object.
    """

    for s_attribute in [
        "m:Ikss:bus1",
        "m:Ikss:bus2",
        "m:Ikss",
    ]:
        f_value = _get_attribute_float(
            o_object,
            s_attribute,
            None,
        )

        if f_value is not None:
            return float(
                f_value,
            )

    return 0.0


def get_distributed_generator_capacity_mva(o_dg):
    """
    Reads installed apparent power of a distributed generator.
    """

    for s_attribute in [
        "sgn",
        "Sn",
        "snom",
        "Srated",
    ]:
        f_value = _get_attribute_float(
            o_dg,
            s_attribute,
            None,
        )

        if f_value is not None:
            return float(
                f_value,
            )

    return 0.0


def get_distributed_generator_capacity_attribute_and_value(o_dg):
    """
    Returns the first available DG capacity attribute and its value.
    This is used for DG capacity randomization.
    """

    for s_attribute in [
        "sgn",
        "Sn",
        "snom",
        "Srated",
    ]:
        f_value = _get_attribute_float(
            o_dg,
            s_attribute,
            None,
        )

        if f_value is not None:
            return s_attribute, float(
                f_value,
            )

    return None, 0.0


def get_unique_objects(l_objects):
    """
    Removes duplicate PowerFactory objects by full name.
    """

    l_unique_objects = []
    s_seen_full_names = set()

    for o_object in l_objects:
        s_full_name = _get_safe_full_name(
            o_object,
        )

        if s_full_name in s_seen_full_names:
            continue

        s_seen_full_names.add(
            s_full_name,
        )

        l_unique_objects.append(
            o_object,
        )

    return l_unique_objects


def get_connected_terminal_for_distributed_generator(o_dg):
    """
    Returns the terminal connected to a distributed generator.
    """

    o_cubicle = _get_attribute_object(
        o_dg,
        "bus1",
    )

    return _get_terminal_from_cubicle(
        o_cubicle,
    )


def get_all_grid_distributed_generators(o_grid):
    """
    Returns all in-service distributed generators inside the selected grid.
    """

    l_all_dg = []

    for s_filter in [
        "*.ElmGenstat",
        "*.ElmPvsys",
    ]:
        try:
            l_objects = o_pf.GetCalcRelevantObjects(
                s_filter,
            ) or []

        except Exception:
            l_objects = []

        for o_dg in l_objects:
            if not _is_object_in_service(
                    o_dg,
            ):
                continue

            o_cubicle = _get_attribute_object(
                o_dg,
                "bus1",
            )

            if get_cubicle_switch_closed_state(
                    o_cubicle,
            ) == 0:
                continue

            o_terminal = _get_terminal_from_cubicle(
                o_cubicle,
            )

            if not _is_object_inside_grid(
                    o_terminal,
                    o_grid,
            ):
                continue

            l_all_dg.append(
                o_dg,
            )

    return get_unique_objects(
        l_all_dg,
    )


def get_original_outserv_states(l_objects):
    """
    Stores original outserv states.
    """

    d_original_states = {}

    for o_object in l_objects:
        try:
            d_original_states[o_object] = int(
                o_object.outserv,
            )
        except Exception:
            pass

    return d_original_states


def restore_original_outserv_states(d_original_states):
    """
    Restores original outserv states.
    """

    for o_object, i_state in d_original_states.items():
        try:
            o_object.outserv = i_state
        except Exception:
            pass


def activate_only_one_distributed_generator(
        l_all_dg,
        o_active_dg,
):
    """
    Switches all DGs in the selected grid out of service except one active DG.
    """

    for o_dg in l_all_dg:
        try:
            if o_dg == o_active_dg:
                o_dg.outserv = 0
            else:
                o_dg.outserv = 1
        except Exception:
            pass


def execute_short_circuit_at_line_fault(
        o_project,
        o_fault_line,
        f_fault_percent,
        f_local_fault_distance_km=None,
):
    """
    Executes one short-circuit calculation at a selected line location.

    The function writes both:
        - percentage-style location attributes
        - km-style faultloc attribute

    This is useful because PowerFactory study cases can differ in which
    attribute is active.
    """

    if o_fault_line is None:
        return False

    o_shc = get_short_circuit_command(
        o_project,
    )

    if o_shc is None:
        _print("⚠️ No ComShc short-circuit command found.")
        return False

    configure_short_circuit_command(
        o_shc,
    )

    _safe_set_attribute(
        o_shc,
        "shcobj",
        o_fault_line,
    )

    # Percentage-style attributes.
    for s_location_attribute in [
        "ppro",
        "relpos",
        "xloc",
        "fltloc",
        "loc_fault",
    ]:
        _safe_set_attribute(
            o_shc,
            s_location_attribute,
            float(f_fault_percent or 0.0),
        )

    # Km-style attribute, used in your older _zone_calculation.py logic.
    if f_local_fault_distance_km is not None:
        _safe_set_attribute(
            o_shc,
            "faultloc",
            float(f_local_fault_distance_km or 0.0),
        )

    try:
        i_status = o_shc.Execute()

        return i_status == 0

    except Exception as o_error:
        _print(
            f"⚠️ Short-circuit execution failed at "
            f"{_get_safe_name(o_fault_line)} "
            f"({f_fault_percent:.2f}%): {o_error}"
        )

        return False


def calculate_zone_infeed_summary_for_turbines(
        o_project,
        o_grid,
        o_start_terminal,
        o_reference_cubicle,
        o_reference_line,
        s_relay_ikss_line_attr,
        l_turbines,
        l_fault_lines,
        f_fault_reach_r_ohm,
        f_fault_reach_x_ohm,
        f_relay_reference_ikss_ka,
        s_zone_name,
):
    """
    Calculates SHC-based infeed correction for one zone.

    Correct logic:
        for each turbine:
            1. isolate the turbine
            2. run SHC at the zone fault point
            3. read turbine Ikss
            4. read relay/reference Ikss
            5. calculate impedance from turbine to fault
            6. correction = impedance_to_fault * Ikss_DG / Ikss_reference

    The zone correction is the sum of all turbine corrections.
    """

    d_summary = {
        "turbines_candidate_count": 0,
        "turbines_candidate_id": "",

        "turbines_considered_count": 0,
        "turbines_considered_id": "",

        "turbines_skipped_count": 0,
        "turbines_skipped_id": "",

        "turbines_skipped_not_on_fault_path_count": 0,
        "turbines_skipped_outside_zone_reach_count": 0,
        "turbines_skipped_no_valid_fault_point_count": 0,
        "turbines_skipped_fault_distance_missing_count": 0,
        "turbines_skipped_fault_not_downstream_count": 0,

        "turbines_total_capacity_mva": 0.0,
        "total_ikss_contribution_ratio": 0.0,
        "max_single_ikss_contribution_ratio": 0.0,
        "infeed_correction_r_ohm": 0.0,
        "infeed_correction_x_ohm": 0.0,
        "relay_reference_ikss_ka": f_relay_reference_ikss_ka,
    }

    l_turbines = get_unique_objects(
        l_turbines,
    )

    d_summary["turbines_candidate_count"] = len(
        l_turbines,
    )

    d_summary["turbines_candidate_id"] = get_object_id_string(
        l_turbines,
    )

    l_skipped_turbines = []

    if not l_turbines:
        return d_summary

    l_path_segments = build_ordered_path_segments(
        o_start_terminal,
        l_fault_lines,
    )

    if not l_path_segments:
        _print(
            f"⚠️ {s_zone_name}: no valid ordered path segments."
        )
        return d_summary

    (
        _,
        _,
        _,
        f_zone_boundary_distance_km,
    ) = select_fault_location_by_reach_impedance(
        l_path_segments=l_path_segments,
        f_target_reach_r_ohm=f_fault_reach_r_ohm,
        f_target_reach_x_ohm=f_fault_reach_x_ohm,
    )

    if f_zone_boundary_distance_km <= 0.0:
        _print(
            f"⚠️ {s_zone_name}: no valid zone boundary distance found."
        )
        return d_summary

    # -------------------------------------------------------------------------
    # Keep only turbines that are actually located on the selected path before
    # the fault point.
    # -------------------------------------------------------------------------
    l_valid_turbine_items = []

    for o_dg in l_turbines:
        o_dg_terminal = get_connected_terminal_for_distributed_generator(
            o_dg,
        )

        f_dg_distance_km = get_terminal_distance_on_ordered_path(
            o_dg_terminal,
            l_path_segments,
        )

        if f_dg_distance_km is None:
            d_summary["turbines_skipped_not_on_fault_path_count"] += 1

            l_skipped_turbines.append(
                o_dg,
            )

            _print(
                f"   {s_zone_name}: skipping {_get_safe_name(o_dg)} "
                f"because its terminal is not on the selected fault path."
            )
            continue

        if f_dg_distance_km > f_zone_boundary_distance_km + 1e-9:
            d_summary["turbines_skipped_outside_zone_reach_count"] += 1

            l_skipped_turbines.append(
                o_dg,
            )

            _print(
                f"   {s_zone_name}: skipping {_get_safe_name(o_dg)} "
                f"because the DG is outside the zone reach."
            )
            continue

        d_fault_context = select_shc_fault_location_for_dg_context(
            o_dg_terminal=o_dg_terminal,
            l_path_segments=l_path_segments,
        )

        if d_fault_context is None:
            d_summary["turbines_skipped_no_valid_fault_point_count"] += 1

            l_skipped_turbines.append(
                o_dg,
            )

            _print(
                f"   {s_zone_name}: skipping {_get_safe_name(o_dg)} "
                f"because no valid DG-specific fault point could be selected."
            )
            continue

        f_fault_distance_km = d_fault_context.get(
            "fault_distance_km",
            None,
        )

        if f_fault_distance_km is None:
            d_summary["turbines_skipped_fault_distance_missing_count"] += 1

            l_skipped_turbines.append(
                o_dg,
            )

            _print(
                f"   {s_zone_name}: skipping {_get_safe_name(o_dg)} "
                f"because the selected fault distance is not available."
            )
            continue

        if f_fault_distance_km <= f_dg_distance_km + 1e-9:
            d_summary["turbines_skipped_fault_not_downstream_count"] += 1

            l_skipped_turbines.append(
                o_dg,
            )

            _print(
                f"   {s_zone_name}: skipping {_get_safe_name(o_dg)} "
                f"because the selected fault is not downstream of the DG."
            )
            continue

        l_valid_turbine_items.append({
            "object": o_dg,
            "terminal": o_dg_terminal,
            "distance_km": f_dg_distance_km,
            "fault_context": d_fault_context,
        })

    d_summary["turbines_skipped_count"] = len(
        get_unique_objects(
            l_skipped_turbines,
        )
    )

    d_summary["turbines_skipped_id"] = get_object_id_string(
        l_skipped_turbines,
    )

    if not l_valid_turbine_items:
        return d_summary

    d_summary["turbines_considered_count"] = len(
        l_valid_turbine_items,
    )

    d_summary["turbines_considered_id"] = get_object_id_string([
        d_item["object"]
        for d_item in l_valid_turbine_items
    ])

    d_summary["turbines_skipped_count"] = len(
        get_unique_objects(
            l_skipped_turbines,
        )
    )

    d_summary["turbines_skipped_id"] = get_object_id_string(
        l_skipped_turbines,
    )

    if (
            d_summary["turbines_candidate_count"]
            != d_summary["turbines_considered_count"]
            + d_summary["turbines_skipped_count"]
    ):
        _print(
            f"⚠️ {s_zone_name}: turbine count mismatch. "
            f"candidate={d_summary['turbines_candidate_count']} | "
            f"considered={d_summary['turbines_considered_count']} | "
            f"skipped={d_summary['turbines_skipped_count']}"
        )

    d_summary["turbines_total_capacity_mva"] = round(
        sum([
            get_distributed_generator_capacity_mva(
                d_item["object"],
            )
            for d_item in l_valid_turbine_items
        ]),
        3,
    )

    l_all_grid_dg = get_all_grid_distributed_generators(
        o_grid,
    )

    d_original_outserv_states = get_original_outserv_states(
        l_all_grid_dg,
    )

    l_single_dg_ratios = []
    f_total_correction_r_ohm = 0.0
    f_total_correction_x_ohm = 0.0
    f_last_reference_ikss_ka = f_relay_reference_ikss_ka

    try:
        for d_item in l_valid_turbine_items:
            o_dg = d_item["object"]
            f_dg_distance_km = d_item["distance_km"]
            d_fault_context = d_item["fault_context"]

            o_fault_line = d_fault_context["fault_line"]
            f_fault_percent = d_fault_context["fault_percent"]
            f_local_fault_distance_km = d_fault_context["local_fault_distance_km"]
            f_fault_distance_km = d_fault_context["fault_distance_km"]
            s_fault_rule = d_fault_context["fault_rule"]

            activate_only_one_distributed_generator(
                l_all_grid_dg,
                o_dg,
            )

            b_success = execute_short_circuit_at_line_fault(
                o_project=o_project,
                o_fault_line=o_fault_line,
                f_fault_percent=f_fault_percent,
                f_local_fault_distance_km=f_local_fault_distance_km,
            )

            if not b_success:
                continue

            f_dg_ikss_ka = get_ikss_value(
                o_dg,
            )

            if f_dg_ikss_ka <= 0.0:
                o_dg_cubicle = _get_attribute_object(
                    o_dg,
                    "bus1",
                )

                f_dg_ikss_ka = get_ikss_value(
                    o_dg_cubicle,
                )

            f_reference_ikss_ka, _ = _read_relay_reference_ikss(
                o_reference_cubicle,
                o_reference_line,
                s_relay_ikss_line_attr,
            )

            if f_reference_ikss_ka <= 0.0:
                f_reference_ikss_ka = f_relay_reference_ikss_ka

            if f_reference_ikss_ka <= 0.0:
                _print(
                    f"⚠️ {s_zone_name}: reference Ikss is zero for "
                    f"{_get_safe_name(o_dg)}."
                )
                continue

            f_last_reference_ikss_ka = f_reference_ikss_ka

            f_ikss_ratio = (
                    f_dg_ikss_ka
                    / f_reference_ikss_ka
            )

            (
                f_impedance_to_fault_r_ohm,
                f_impedance_to_fault_x_ohm,
            ) = calculate_path_impedance_between_distances(
                l_path_segments,
                f_dg_distance_km,
                f_fault_distance_km,
            )

            f_dg_correction_r_ohm = (
                    f_impedance_to_fault_r_ohm
                    * f_ikss_ratio
            )

            f_dg_correction_x_ohm = (
                    f_impedance_to_fault_x_ohm
                    * f_ikss_ratio
            )

            f_total_correction_r_ohm += f_dg_correction_r_ohm
            f_total_correction_x_ohm += f_dg_correction_x_ohm

            l_single_dg_ratios.append(
                f_ikss_ratio,
            )

            _print(
                f"   {s_zone_name}: "
                f"DG={_get_safe_name(o_dg)} | "
                f"fault_rule={s_fault_rule} | "
                f"fault_line={_get_safe_name(o_fault_line)} | "
                f"DG_dist={f_dg_distance_km:.3f} km | "
                f"fault_dist={f_fault_distance_km:.3f} km | "
                f"Ikss_DG={f_dg_ikss_ka:.3f} kA | "
                f"Ikss_ref={f_reference_ikss_ka:.3f} kA | "
                f"ratio={f_ikss_ratio:.3f} | "
                f"Z_to_fault=({f_impedance_to_fault_r_ohm:.3f}, "
                f"{f_impedance_to_fault_x_ohm:.3f}) | "
                f"corr=({f_dg_correction_r_ohm:.3f}, "
                f"{f_dg_correction_x_ohm:.3f})"
            )

    finally:
        restore_original_outserv_states(
            d_original_outserv_states,
        )

    d_summary["relay_reference_ikss_ka"] = round(
        f_last_reference_ikss_ka,
        3,
    )

    d_summary["total_ikss_contribution_ratio"] = round(
        sum(l_single_dg_ratios),
        3,
    )

    d_summary["max_single_ikss_contribution_ratio"] = round(
        max(l_single_dg_ratios) if l_single_dg_ratios else 0.0,
        3,
    )

    d_summary["infeed_correction_r_ohm"] = round(
        f_total_correction_r_ohm,
        3,
    )

    d_summary["infeed_correction_x_ohm"] = round(
        f_total_correction_x_ohm,
        3,
    )

    return d_summary


def build_ordered_path_segments(
        o_start_terminal,
        l_path_lines,
):
    """
    Builds ordered path segments from relay busbar to the selected fault-path end.

    Each segment stores:
        - line object
        - start terminal
        - end terminal
        - cumulative start distance
        - cumulative end distance
        - line R/X
        - line length
    """

    l_segments = []

    if o_start_terminal is None or not l_path_lines:
        return l_segments

    o_current_terminal = o_start_terminal
    f_cumulative_distance_km = 0.0

    for o_line in l_path_lines:
        o_next_terminal = _get_opposite_terminal(
            o_line,
            o_current_terminal,
        )

        if o_next_terminal is None:
            break

        f_line_length_km = get_line_length_value(
            o_line,
        )

        f_line_r_ohm, f_line_x_ohm = _get_line_impedance(
            o_line,
        )

        l_segments.append({
            "line": o_line,
            "start_terminal": o_current_terminal,
            "end_terminal": o_next_terminal,
            "start_distance_km": f_cumulative_distance_km,
            "end_distance_km": f_cumulative_distance_km + f_line_length_km,
            "length_km": f_line_length_km,
            "r_ohm": f_line_r_ohm,
            "x_ohm": f_line_x_ohm,
        })

        f_cumulative_distance_km += f_line_length_km
        o_current_terminal = o_next_terminal

    return l_segments


def get_terminal_distance_on_ordered_path(
        o_terminal,
        l_path_segments,
):
    """
    Returns terminal distance from the relay/start terminal along the ordered path.
    """

    if o_terminal is None:
        return None

    for d_segment in l_path_segments:
        if o_terminal == d_segment["start_terminal"]:
            return d_segment["start_distance_km"]

        if o_terminal == d_segment["end_terminal"]:
            return d_segment["end_distance_km"]

    return None


def select_fault_location_on_ordered_path(
        l_path_segments,
        f_fault_path_fraction,
):
    """
    Selects the fault line, local fault percent, local fault km, and absolute
    distance from relay/start terminal.
    """

    if not l_path_segments:
        return None, 0.0, 0.0, 0.0

    f_total_path_length_km = l_path_segments[-1]["end_distance_km"]

    if f_total_path_length_km <= 0.0:
        return l_path_segments[0]["line"], 0.0, 0.0, 0.0

    f_fault_path_fraction = max(
        0.0,
        min(
            1.0,
            float(f_fault_path_fraction or 0.0),
        ),
    )

    f_fault_distance_km = f_fault_path_fraction * f_total_path_length_km

    for d_segment in l_path_segments:
        if (
                d_segment["start_distance_km"]
                <= f_fault_distance_km
                <= d_segment["end_distance_km"]
        ):
            f_local_fault_distance_km = (
                    f_fault_distance_km
                    - d_segment["start_distance_km"]
            )

            if d_segment["length_km"] > 0.0:
                f_fault_percent = (
                        100.0
                        * f_local_fault_distance_km
                        / d_segment["length_km"]
                )
            else:
                f_fault_percent = 0.0

            return (
                d_segment["line"],
                f_fault_percent,
                f_local_fault_distance_km,
                f_fault_distance_km,
            )

    d_last_segment = l_path_segments[-1]

    return (
        d_last_segment["line"],
        100.0,
        d_last_segment["length_km"],
        f_total_path_length_km,
    )


def calculate_path_impedance_between_distances(
        l_path_segments,
        f_from_distance_km,
        f_to_distance_km,
):
    """
    Calculates R/X impedance between two distances on the ordered path.
    """

    if f_from_distance_km is None or f_to_distance_km is None:
        return 0.0, 0.0

    f_start_distance_km = min(
        f_from_distance_km,
        f_to_distance_km,
    )

    f_end_distance_km = max(
        f_from_distance_km,
        f_to_distance_km,
    )

    f_total_r_ohm = 0.0
    f_total_x_ohm = 0.0

    for d_segment in l_path_segments:
        f_segment_start = d_segment["start_distance_km"]
        f_segment_end = d_segment["end_distance_km"]
        f_segment_length = d_segment["length_km"]

        f_overlap_start = max(
            f_start_distance_km,
            f_segment_start,
        )

        f_overlap_end = min(
            f_end_distance_km,
            f_segment_end,
        )

        if f_overlap_end <= f_overlap_start:
            continue

        if f_segment_length <= 0.0:
            continue

        f_overlap_fraction = (
                (f_overlap_end - f_overlap_start)
                / f_segment_length
        )

        f_total_r_ohm += d_segment["r_ohm"] * f_overlap_fraction
        f_total_x_ohm += d_segment["x_ohm"] * f_overlap_fraction

    return round(f_total_r_ohm, 3), round(f_total_x_ohm, 3)


def select_fault_location_by_reach_impedance(
        l_path_segments,
        f_target_reach_r_ohm,
        f_target_reach_x_ohm,
):
    """
    Selects the fault location based on the actual impedance reach.

    The distance is found by walking along the ordered path until the cumulative
    X impedance reaches the target zone X reach.

    X is used as the primary reach coordinate because distance relay reach is
    normally dominated by reactance and X is monotonic along the line path.
    """

    if not l_path_segments:
        return None, 0.0, 0.0, 0.0

    f_target_reach_x_ohm = max(
        0.0,
        float(f_target_reach_x_ohm or 0.0),
    )

    f_cumulative_x_ohm = 0.0

    for d_segment in l_path_segments:
        f_segment_x_ohm = float(
            d_segment.get(
                "x_ohm",
                0.0,
            ) or 0.0
        )

        f_segment_length_km = float(
            d_segment.get(
                "length_km",
                0.0,
            ) or 0.0
        )

        f_segment_start_distance_km = float(
            d_segment.get(
                "start_distance_km",
                0.0,
            ) or 0.0
        )

        # If this segment contains the reach boundary, interpolate inside it.
        if f_segment_x_ohm > 0.0 and (
                f_cumulative_x_ohm + f_segment_x_ohm
                >= f_target_reach_x_ohm
        ):
            f_remaining_x_ohm = (
                    f_target_reach_x_ohm
                    - f_cumulative_x_ohm
            )

            f_fraction_on_segment = (
                    f_remaining_x_ohm
                    / f_segment_x_ohm
            )

            f_fraction_on_segment = max(
                0.0,
                min(
                    1.0,
                    f_fraction_on_segment,
                ),
            )

            f_local_fault_distance_km = (
                    f_fraction_on_segment
                    * f_segment_length_km
            )

            f_fault_percent = (
                    100.0
                    * f_fraction_on_segment
            )

            f_fault_distance_km = (
                    f_segment_start_distance_km
                    + f_local_fault_distance_km
            )

            return (
                d_segment["line"],
                f_fault_percent,
                f_local_fault_distance_km,
                f_fault_distance_km,
            )

        f_cumulative_x_ohm += f_segment_x_ohm

    # If target reach is beyond the available path, clamp to the path end.
    d_last_segment = l_path_segments[-1]

    return (
        d_last_segment["line"],
        100.0,
        d_last_segment.get("length_km", 0.0),
        d_last_segment.get("end_distance_km", 0.0),
    )


def select_shc_fault_location_for_dg_context(
        o_dg_terminal,
        l_path_segments,
):
    """
    Selects SHC fault location using the DG-location rule.

    Rule:
        - If DG is on a busbar:
              fault at 50% of the line below that busbar.
        - If DG is on a junction node:
              fault at 50% of the last section of the multi-section line.

    Important:
        This function does NOT check whether the fault point is inside the zone.
        The zone check must be applied to the DG location, not to the fault point.
    """

    if o_dg_terminal is None:
        return None

    if not l_path_segments:
        return None

    # -------------------------------------------------------------------------
    # Case 1: DG connected to a busbar.
    # Fault at 50% of the first downstream line starting from that busbar.
    # -------------------------------------------------------------------------
    if get_terminal_is_busbar(
            o_dg_terminal,
    ):
        l_downstream_branch_segments = []

        b_collecting_branch = False

        for d_segment in l_path_segments:
            if not b_collecting_branch:
                if d_segment.get("start_terminal") != o_dg_terminal:
                    continue

                b_collecting_branch = True

            l_downstream_branch_segments.append(
                d_segment,
            )

            o_end_terminal = d_segment.get(
                "end_terminal",
            )

            if (
                    o_end_terminal != o_dg_terminal
                    and get_terminal_is_busbar(o_end_terminal)
            ):
                break

        if not l_downstream_branch_segments:
            return None

        f_total_branch_length_km = sum([
            float(d_segment.get("length_km", 0.0) or 0.0)
            for d_segment in l_downstream_branch_segments
        ])

        if f_total_branch_length_km <= 0.0:
            return None

        f_midpoint_distance_inside_branch_km = (
                0.5
                * f_total_branch_length_km
        )

        f_accumulated_length_km = 0.0

        for d_segment in l_downstream_branch_segments:
            f_segment_length_km = float(
                d_segment.get(
                    "length_km",
                    0.0,
                ) or 0.0
            )

            if (
                    f_accumulated_length_km
                    + f_segment_length_km
                    >= f_midpoint_distance_inside_branch_km
            ):
                f_local_fault_distance_km = (
                        f_midpoint_distance_inside_branch_km
                        - f_accumulated_length_km
                )

                if f_segment_length_km > 0.0:
                    f_fault_percent = (
                            100.0
                            * f_local_fault_distance_km
                            / f_segment_length_km
                    )
                else:
                    f_fault_percent = 0.0

                f_fault_distance_km = (
                        float(
                            d_segment.get(
                                "start_distance_km",
                                0.0,
                            ) or 0.0
                        )
                        + f_local_fault_distance_km
                )

                return {
                    "fault_line": d_segment.get("line"),
                    "fault_percent": f_fault_percent,
                    "local_fault_distance_km": f_local_fault_distance_km,
                    "fault_distance_km": f_fault_distance_km,
                    "fault_rule": "busbar_dg_fault_at_50_percent_of_full_multisection_branch_below",
                }

            f_accumulated_length_km += f_segment_length_km

        d_last_segment = l_downstream_branch_segments[-1]

        return {
            "fault_line": d_last_segment.get("line"),
            "fault_percent": 100.0,
            "local_fault_distance_km": float(
                d_last_segment.get(
                    "length_km",
                    0.0,
                ) or 0.0
            ),
            "fault_distance_km": float(
                d_last_segment.get(
                    "end_distance_km",
                    0.0,
                ) or 0.0
            ),
            "fault_rule": "busbar_dg_fault_at_50_percent_of_full_multisection_branch_below",
        }

    # -------------------------------------------------------------------------
    # Case 2: DG connected to a junction node.
    # Fault at 50% of the last section of the multi-section line.
    # -------------------------------------------------------------------------
    if get_terminal_is_junction_node(
            o_dg_terminal,
    ):
        i_start_segment_index = None

        for i_index, d_segment in enumerate(l_path_segments):
            if d_segment.get("start_terminal") == o_dg_terminal:
                i_start_segment_index = i_index
                break

            if d_segment.get("end_terminal") == o_dg_terminal:
                i_start_segment_index = i_index + 1
                break

        if i_start_segment_index is None:
            return None

        d_last_section = None

        for d_segment in l_path_segments[i_start_segment_index:]:
            d_last_section = d_segment

            o_end_terminal = d_segment.get(
                "end_terminal",
            )

            if get_terminal_is_busbar(
                    o_end_terminal,
            ):
                break

        if d_last_section is None:
            return None

        f_segment_length_km = float(
            d_last_section.get(
                "length_km",
                0.0,
            ) or 0.0
        )

        f_local_fault_distance_km = 0.5 * f_segment_length_km

        f_fault_distance_km = (
            float(
                d_last_section.get(
                    "start_distance_km",
                    0.0,
                ) or 0.0
            )
            + f_local_fault_distance_km
        )

        return {
            "fault_line": d_last_section.get("line"),
            "fault_percent": 50.0,
            "local_fault_distance_km": f_local_fault_distance_km,
            "fault_distance_km": f_fault_distance_km,
            "fault_rule": "junction_dg_fault_at_50_percent_of_last_multisection_section",
        }

    return None


# =============================================================================
# Basic Object Utility Functions
# =============================================================================

def get_boolean_value(b_value):
    return 1 if bool(b_value) else 0


def get_line_terminals(o_line):
    o_bus1_cubicle = _get_attribute_object(o_line, "bus1")
    o_bus2_cubicle = _get_attribute_object(o_line, "bus2")

    o_terminal_1 = _get_terminal_from_cubicle(o_bus1_cubicle)
    o_terminal_2 = _get_terminal_from_cubicle(o_bus2_cubicle)

    return o_terminal_1, o_terminal_2, o_bus1_cubicle, o_bus2_cubicle


def get_cubicle_for_line_at_terminal(o_line, o_terminal):
    o_terminal_1, o_terminal_2, o_cubicle_1, o_cubicle_2 = get_line_terminals(o_line)

    if o_terminal_1 == o_terminal:
        return o_cubicle_1

    if o_terminal_2 == o_terminal:
        return o_cubicle_2

    return None


def get_cubicle_switch_closed_state(
        o_cubicle,
):
    """
    Returns the closed/open state of the first StaSwitch inside a cubicle.

    Return:
        1 = closed / usable
        0 = open / not usable

    If the cubicle has no StaSwitch object, it is treated as closed.
    """

    if o_cubicle is None:
        return 0

    o_switch = get_first_cubicle_switch(
        o_cubicle,
    )

    if o_switch is None:
        return 1

    try:
        return 1 if int(o_switch.on_off) == 1 else 0

    except Exception:
        try:
            return 1 if int(o_switch.GetAttribute("on_off")) == 1 else 0

        except Exception:
            return 1


def get_line_switch_state_summary(
        o_line,
):
    """
    Returns switch-state information for both line-end cubicles.
    """

    (
        _,
        _,
        o_cubicle_1,
        o_cubicle_2,
    ) = get_line_terminals(
        o_line,
    )

    i_bus1_switch_closed = get_cubicle_switch_closed_state(
        o_cubicle_1,
    )

    i_bus2_switch_closed = get_cubicle_switch_closed_state(
        o_cubicle_2,
    )

    i_line_switch_closed = (
        1
        if i_bus1_switch_closed == 1 and i_bus2_switch_closed == 1
        else 0
    )

    return {
        "bus1_switch_closed": i_bus1_switch_closed,
        "bus2_switch_closed": i_bus2_switch_closed,
        "line_switch_closed": i_line_switch_closed,
    }


def get_line_is_available_for_topology(
        o_line,
):
    """
    A line is available for corridor/topology traversal only if:
        - the ElmLne object itself is in service
        - both line-end cubicle switches are closed

    This prevents open-switched lines from appearing in the NetworkX graph.
    """

    if o_line is None:
        return False

    if not _is_object_in_service(
            o_line,
    ):
        return False

    d_switch_state = get_line_switch_state_summary(
        o_line,
    )

    return int(
        d_switch_state.get(
            "line_switch_closed",
            0,
        )
    ) == 1


def get_object_connection_cubicles(
        o_object,
):
    """
    Returns all connection cubicles of a PowerFactory component.

    Covered object types:
        - Lines: bus1, bus2
        - DGs / loads: bus1
        - Transformers: bushv, busmv, buslv, bus1, bus2, bus3
    """

    l_cubicles = []

    for s_attribute in [
        "bus1",
        "bus2",
        "bus3",
        "bushv",
        "busmv",
        "buslv",
    ]:
        o_cubicle = _get_attribute_object(
            o_object,
            s_attribute,
        )

        if o_cubicle is None:
            continue

        l_cubicles.append(
            o_cubicle,
        )

    return get_unique_objects(
        l_cubicles,
    )


def get_switch_state_component_group(
        o_object,
):
    """
    Returns a simple component group name for logging.
    """

    s_class_name = _get_safe_class_name(
        o_object,
    )

    if s_class_name == "ElmLne":
        return "line"

    if s_class_name in [
        "ElmGenstat",
        "ElmPvsys",
    ]:
        return "dg"

    if s_class_name in [
        "ElmLod",
        "ElmLodlv",
    ]:
        return "load"

    if s_class_name in [
        "ElmTr2",
        "ElmTr3",
    ]:
        return "transformer"

    return "other"


def get_component_has_open_connection_switch(
        o_object,
):
    """
    Returns True if at least one connection cubicle of the component has
    an open switch.
    """

    for o_cubicle in get_object_connection_cubicles(
            o_object,
    ):
        if get_cubicle_switch_closed_state(
                o_cubicle,
        ) == 0:
            return True

    return False


def get_switch_state_outserv_controlled_objects(
        o_grid,
):
    """
    Returns components whose outserv state may be temporarily controlled
    by switch-state scenarios.
    """

    l_component_objects = []

    for s_filter in [
        "*.ElmLne",
        "*.ElmGenstat",
        "*.ElmPvsys",
        "*.ElmLod",
        "*.ElmLodlv",
        "*.ElmTr2",
        "*.ElmTr3",
    ]:
        try:
            l_objects = o_pf.GetCalcRelevantObjects(
                s_filter,
            ) or []

        except Exception:
            l_objects = []

        for o_object in l_objects:
            l_cubicles = get_object_connection_cubicles(
                o_object,
            )

            if not l_cubicles:
                continue

            b_inside_grid = False

            for o_cubicle in l_cubicles:
                o_terminal = _get_terminal_from_cubicle(
                    o_cubicle,
                )

                if _is_object_inside_grid(
                        o_terminal,
                        o_grid,
                ):
                    b_inside_grid = True
                    break

            if not b_inside_grid:
                continue

            l_component_objects.append(
                o_object,
            )

    l_component_objects = get_unique_objects(
        l_component_objects,
    )

    d_count_by_class = defaultdict(
        int,
    )

    for o_object in l_component_objects:
        d_count_by_class[
            _get_safe_class_name(
                o_object,
            )
        ] += 1

    _print(
        f"✅ Switch-state outserv-controlled components found: "
        f"{len(l_component_objects)}"
    )

    if b_enable_debug:
        for s_class_name, i_count in sorted(
                d_count_by_class.items(),
        ):
            _print(
                f"   {s_class_name}: {i_count}"
            )

    return l_component_objects


def apply_outserv_for_components_behind_open_switches(
        l_component_objects,
        d_original_outserv_states,
        s_switch_state_short_id,
        s_switch_state_config_id,
        s_scenario_id,
        b_collect_log_rows=True,
):
    """
    Temporarily sets components behind open cubicle switches out of service.

    Rule:
        if original outserv == 0
        and at least one connection cubicle switch is open
        then set component.outserv = 1

    Original outserv states must be restored by the caller.
    """

    d_summary = {
        "switch_state_forced_outserv_component_count": 0,
        "switch_state_forced_outserv_line_count": 0,
        "switch_state_forced_outserv_dg_count": 0,
        "switch_state_forced_outserv_load_count": 0,
        "switch_state_forced_outserv_transformer_count": 0,
        "switch_state_forced_outserv_other_count": 0,
    }

    l_log_rows = []

    if not b_enable_switch_state_outserv_derivation:
        return d_summary, l_log_rows

    for o_object in l_component_objects:
        i_original_outserv = int(
            d_original_outserv_states.get(
                o_object,
                _get_attribute_int(
                    o_object,
                    "outserv",
                    0,
                ),
            )
            or 0
        )

        # Do not change components that were already out of service originally.
        if i_original_outserv != 0:
            continue

        l_open_cubicles = []

        for o_cubicle in get_object_connection_cubicles(
                o_object,
        ):
            if get_cubicle_switch_closed_state(
                    o_cubicle,
            ) == 0:
                l_open_cubicles.append(
                    o_cubicle,
                )

        if not l_open_cubicles:
            continue

        b_set_success = _safe_set_attribute(
            o_object,
            "outserv",
            1,
        )

        if not b_set_success:
            continue

        s_component_group = get_switch_state_component_group(
            o_object,
        )

        d_summary["switch_state_forced_outserv_component_count"] += 1

        if s_component_group == "line":
            d_summary["switch_state_forced_outserv_line_count"] += 1

        elif s_component_group == "dg":
            d_summary["switch_state_forced_outserv_dg_count"] += 1

        elif s_component_group == "load":
            d_summary["switch_state_forced_outserv_load_count"] += 1

        elif s_component_group == "transformer":
            d_summary["switch_state_forced_outserv_transformer_count"] += 1

        else:
            d_summary["switch_state_forced_outserv_other_count"] += 1

        if b_collect_log_rows:
            l_log_rows.append({
                "switch_state_short_id": s_switch_state_short_id,
                "switch_state_config_id": s_switch_state_config_id,
                "scenario_id": s_scenario_id,
                "component_id": _get_safe_name(
                    o_object,
                ),
                "component_full_name": _get_safe_full_name(
                    o_object,
                ),
                "component_class": _get_safe_class_name(
                    o_object,
                ),
                "component_group": s_component_group,
                "original_outserv": i_original_outserv,
                "temporary_outserv": 1,
                "open_cubicle_count": len(
                    l_open_cubicles,
                ),
                "open_cubicle_id": "; ".join([
                    _get_safe_name(
                        o_cubicle,
                    )
                    for o_cubicle in l_open_cubicles
                ]),
                "open_cubicle_full_name": "; ".join([
                    _get_safe_full_name(
                        o_cubicle,
                    )
                    for o_cubicle in l_open_cubicles
                ]),
            })

    return d_summary, l_log_rows


def get_line_length_value(o_line):
    return _get_attribute_float(o_line, "dline", 0.0)


def store_original_line_length_impedance_state(l_lines):
    """
    Stores original dline, R1, and X1 values before randomization.

    This allows every scenario to start from the same base grid.
    """

    d_original_state = {}

    for o_line in l_lines:
        s_line_key = _get_safe_full_name(
            o_line,
        )

        d_original_state[s_line_key] = {
            "object": o_line,
            "line_name": _get_safe_name(o_line),
            "dline": _get_attribute_float(o_line, "dline", 0.0),
            "R1": _get_attribute_float(o_line, "R1", 0.0),
            "X1": _get_attribute_float(o_line, "X1", 0.0),
        }

    return d_original_state


def restore_original_line_length_impedance_state(d_original_state):
    """
    Restores original dline, R1, and X1 values.
    """

    for _, d_line_state in d_original_state.items():
        o_line = d_line_state.get(
            "object",
            None,
        )

        if o_line is None:
            continue

        _safe_set_attribute(
            o_line,
            "dline",
            d_line_state.get(
                "dline",
                0.0,
            ),
        )

        _safe_set_attribute(
            o_line,
            "R1",
            d_line_state.get(
                "R1",
                0.0,
            ),
        )

        _safe_set_attribute(
            o_line,
            "X1",
            d_line_state.get(
                "X1",
                0.0,
            ),
        )


def apply_random_line_length_scenario(
        d_original_state,
        s_scenario_id,
        i_random_seed,
        f_scale_min,
        f_scale_max,
):
    """
    Applies one randomized line-length scenario.

    For each line:
        dline_new = dline_original * factor
        R1_new    = R1_original * factor
        X1_new    = X1_original * factor

    Returns a log of the applied changes.
    """

    o_random = random.Random(
        i_random_seed,
    )

    l_randomization_log_rows = []

    for _, d_line_state in d_original_state.items():
        o_line = d_line_state.get(
            "object",
            None,
        )

        if o_line is None:
            continue

        f_scale_factor = o_random.uniform(
            f_scale_min,
            f_scale_max,
        )

        f_original_length_km = d_line_state.get(
            "dline",
            0.0,
        )

        f_original_r_ohm = d_line_state.get(
            "R1",
            0.0,
        )

        f_original_x_ohm = d_line_state.get(
            "X1",
            0.0,
        )

        f_new_length_km = f_original_length_km * f_scale_factor
        f_new_r_ohm = f_original_r_ohm * f_scale_factor
        f_new_x_ohm = f_original_x_ohm * f_scale_factor

        b_dline_set = _safe_set_attribute(
            o_line,
            "dline",
            f_new_length_km,
        )

        b_r1_set = _safe_set_attribute(
            o_line,
            "R1",
            f_new_r_ohm,
        )

        b_x1_set = _safe_set_attribute(
            o_line,
            "X1",
            f_new_x_ohm,
        )

        f_actual_length_km = _get_attribute_float(
            o_line,
            "dline",
            0.0,
        )

        f_actual_r_ohm = _get_attribute_float(
            o_line,
            "R1",
            0.0,
        )

        f_actual_x_ohm = _get_attribute_float(
            o_line,
            "X1",
            0.0,
        )

        l_randomization_log_rows.append({
            "scenario_id": s_scenario_id,
            "random_seed": i_random_seed,
            "line_id": d_line_state.get(
                "line_name",
                "",
            ),

            "scale_factor": round(
                f_scale_factor,
                6,
            ),

            "set_dline_success": int(
                b_dline_set,
            ),
            "set_r1_success": int(
                b_r1_set,
            ),
            "set_x1_success": int(
                b_x1_set,
            ),

            "original_length_km": round(
                f_original_length_km,
                6,
            ),
            "randomized_length_km": round(
                f_new_length_km,
                6,
            ),
            "actual_length_km": round(
                f_actual_length_km,
                6,
            ),
            "length_set_error_km": round(
                f_actual_length_km - f_new_length_km,
                9,
            ),

            "original_r_ohm": round(
                f_original_r_ohm,
                6,
            ),
            "randomized_r_ohm": round(
                f_new_r_ohm,
                6,
            ),
            "actual_r_ohm": round(
                f_actual_r_ohm,
                6,
            ),
            "r_set_error_ohm": round(
                f_actual_r_ohm - f_new_r_ohm,
                9,
            ),

            "original_x_ohm": round(
                f_original_x_ohm,
                6,
            ),
            "randomized_x_ohm": round(
                f_new_x_ohm,
                6,
            ),
            "actual_x_ohm": round(
                f_actual_x_ohm,
                6,
            ),
            "x_set_error_ohm": round(
                f_actual_x_ohm - f_new_x_ohm,
                9,
            ),
        })

    return l_randomization_log_rows


def get_randomizable_grid_lines(o_grid):
    """
    Returns in-service line objects inside the selected grid.

    These are the lines whose length/R/X will be randomized.
    """

    l_randomizable_lines = []

    try:
        l_all_lines = o_pf.GetCalcRelevantObjects(
            "*.ElmLne",
        ) or []

    except Exception:
        l_all_lines = []

    for o_line in l_all_lines:
        if not get_line_is_available_for_topology(
                o_line,
        ):
            continue

        o_terminal_1, o_terminal_2, _, _ = get_line_terminals(
            o_line,
        )

        if o_terminal_1 is None or o_terminal_2 is None:
            continue

        if (
                not _is_object_inside_grid(o_terminal_1, o_grid)
                or not _is_object_inside_grid(o_terminal_2, o_grid)
        ):
            continue

        f_length_km = get_line_length_value(
            o_line,
        )

        if f_length_km <= 0.0:
            continue

        l_randomizable_lines.append(
            o_line,
        )

    return get_unique_objects(
        l_randomizable_lines,
    )


def get_randomizable_grid_distributed_generators(o_grid):
    """
    Returns in-service DGs inside the selected grid whose capacity can be randomized.
    """

    l_all_dg = get_all_grid_distributed_generators(
        o_grid,
    )

    l_randomizable_dg = []

    for o_dg in l_all_dg:
        (
            s_capacity_attribute,
            f_capacity_mva,
        ) = get_distributed_generator_capacity_attribute_and_value(
            o_dg,
        )

        if s_capacity_attribute is None:
            continue

        if f_capacity_mva <= 0.0:
            continue

        l_randomizable_dg.append(
            o_dg,
        )

    return get_unique_objects(
        l_randomizable_dg,
    )


def store_original_dg_capacity_state(l_dg):
    """
    Stores original DG capacity values before randomization.
    """

    d_original_state = {}

    for o_dg in l_dg:
        s_dg_key = _get_safe_full_name(
            o_dg,
        )

        (
            s_capacity_attribute,
            f_capacity_mva,
        ) = get_distributed_generator_capacity_attribute_and_value(
            o_dg,
        )

        d_original_state[s_dg_key] = {
            "object": o_dg,
            "dg_name": _get_safe_name(
                o_dg,
            ),
            "capacity_attribute": s_capacity_attribute,
            "capacity_mva": f_capacity_mva,
        }

    return d_original_state


def restore_original_dg_capacity_state(d_original_state):
    """
    Restores original DG capacity values.
    """

    for _, d_dg_state in d_original_state.items():
        o_dg = d_dg_state.get(
            "object",
            None,
        )

        s_capacity_attribute = d_dg_state.get(
            "capacity_attribute",
            None,
        )

        if o_dg is None or s_capacity_attribute is None:
            continue

        _safe_set_attribute(
            o_dg,
            s_capacity_attribute,
            d_dg_state.get(
                "capacity_mva",
                0.0,
            ),
        )


def apply_random_dg_capacity_scenario(
        d_original_state,
        s_scenario_id,
        i_random_seed,
        f_scale_min,
        f_scale_max,
):
    """
    Applies one randomized DG capacity scenario.

    For each DG:
        capacity_new = capacity_original * factor

    Returns a log of intended and actual applied DG capacities.
    """

    o_random = random.Random(
        i_random_seed,
    )

    l_randomization_log_rows = []

    for _, d_dg_state in d_original_state.items():
        o_dg = d_dg_state.get(
            "object",
            None,
        )

        s_capacity_attribute = d_dg_state.get(
            "capacity_attribute",
            None,
        )

        if o_dg is None or s_capacity_attribute is None:
            continue

        f_scale_factor = o_random.uniform(
            f_scale_min,
            f_scale_max,
        )

        f_original_capacity_mva = d_dg_state.get(
            "capacity_mva",
            0.0,
        )

        f_new_capacity_mva = (
                f_original_capacity_mva
                * f_scale_factor
        )

        b_capacity_set = _safe_set_attribute(
            o_dg,
            s_capacity_attribute,
            f_new_capacity_mva,
        )

        f_actual_capacity_mva = _get_attribute_float(
            o_dg,
            s_capacity_attribute,
            0.0,
        )

        l_randomization_log_rows.append({
            "scenario_id": s_scenario_id,
            "random_seed": i_random_seed,
            "dg_id": d_dg_state.get(
                "dg_name",
                "",
            ),
            "capacity_attribute": s_capacity_attribute,
            "scale_factor": round(
                f_scale_factor,
                6,
            ),
            "set_capacity_success": int(
                b_capacity_set,
            ),
            "original_capacity_mva": round(
                f_original_capacity_mva,
                6,
            ),
            "randomized_capacity_mva": round(
                f_new_capacity_mva,
                6,
            ),
            "actual_capacity_mva": round(
                f_actual_capacity_mva,
                6,
            ),
            "capacity_set_error_mva": round(
                f_actual_capacity_mva - f_new_capacity_mva,
                9,
            ),
        })

    return l_randomization_log_rows


def get_terminal_is_junction_node(o_terminal):
    return _get_attribute_int(o_terminal, "iUsage", 0) == 1


def get_terminal_is_busbar(o_terminal):
    return _get_attribute_int(o_terminal, "iUsage", 0) == 0


def get_terminal_connected_elements(o_terminal):
    try:
        return list(o_terminal.GetConnectedElements())

    except Exception:
        return []


def get_terminal_connected_lines(o_terminal):
    l_lines = []

    for o_element in get_terminal_connected_elements(o_terminal):
        if (
                _get_safe_class_name(o_element) == "ElmLne"
                and get_line_is_available_for_topology(o_element)
        ):
            l_lines.append(o_element)

    return l_lines


def get_terminal_connected_loads(o_terminal):
    l_loads = []

    for o_element in get_terminal_connected_elements(o_terminal):
        if (
                _get_safe_class_name(o_element) in ["ElmLod", "ElmLodlv"]
                and _is_object_in_service(o_element)
        ):
            l_loads.append(o_element)

    return l_loads


def get_terminal_connected_generators(o_terminal):
    l_generators = []

    for o_element in get_terminal_connected_elements(o_terminal):
        if (
                _get_safe_class_name(o_element) in ["ElmSym", "ElmGenstat", "ElmPvsys"]
                and _is_object_in_service(o_element)
        ):
            l_generators.append(o_element)

    return l_generators


def get_terminal_connected_distributed_generators(o_terminal):
    l_generators = []

    for o_element in get_terminal_connected_elements(o_terminal):
        if (
                _get_safe_class_name(o_element) in ["ElmGenstat", "ElmPvsys"]
                and _is_object_in_service(o_element)
        ):
            l_generators.append(o_element)

    return l_generators


def get_terminal_has_load(o_terminal):
    return len(get_terminal_connected_loads(o_terminal)) > 0


def get_terminal_has_distributed_generation(o_terminal):
    return len(get_terminal_connected_distributed_generators(o_terminal)) > 0


def get_terminal_load_p_mw(o_terminal):
    f_total_load_p_mw = 0.0

    for o_load in get_terminal_connected_loads(o_terminal):
        f_p = _get_attribute_float(o_load, "plini", None)

        if f_p is None:
            f_p = _get_attribute_float(o_load, "pgini", None)

        if f_p is None:
            f_p = 0.0

        f_total_load_p_mw += f_p

    return f_total_load_p_mw


def get_object_is_relay(o_object):
    s_class_name = _get_safe_class_name(o_object).lower()
    s_object_name = _get_safe_name(o_object).lower()

    l_relay_markers = [
        "relay",
        "rel",
        "reldis",
        "distance",
        "reloc",
        "elmrelay",
    ]

    for s_marker in l_relay_markers:
        if s_marker in s_class_name or s_marker in s_object_name:
            return True

    return False


def get_terminal_connected_relays(o_terminal):
    l_relays = []

    for o_element in get_terminal_connected_elements(o_terminal):
        if get_object_is_relay(o_element):
            l_relays.append(o_element)

    for o_line in get_terminal_connected_lines(o_terminal):
        o_terminal_1, o_terminal_2, o_cubicle_1, o_cubicle_2 = get_line_terminals(o_line)

        o_cubicle = None

        if o_terminal_1 == o_terminal:
            o_cubicle = o_cubicle_1

        elif o_terminal_2 == o_terminal:
            o_cubicle = o_cubicle_2

        if o_cubicle is None:
            continue

        try:
            l_cubicle_contents = o_cubicle.GetContents("*", 1)
        except Exception:
            l_cubicle_contents = []

        for o_element in l_cubicle_contents:
            if get_object_is_relay(o_element):
                l_relays.append(o_element)

    l_unique_relays = []
    l_seen_names = []

    for o_relay in l_relays:
        s_full_name = _get_safe_full_name(o_relay)

        if s_full_name not in l_seen_names:
            l_seen_names.append(s_full_name)
            l_unique_relays.append(o_relay)

    return l_unique_relays


def get_terminal_has_relay(o_terminal):
    return len(get_terminal_connected_relays(o_terminal)) > 0


def get_relay_id_from_terminal_and_line(o_terminal, o_line):
    l_relays = get_terminal_connected_relays(o_terminal)

    if l_relays:
        return _get_safe_name(l_relays[0])

    return f"relay_node_{_get_safe_name(o_terminal)}_line_{_get_safe_name(o_line)}"


def get_line_connection_type(o_from_terminal, o_to_terminal):
    b_from_busbar = get_terminal_is_busbar(o_from_terminal)
    b_from_junction = get_terminal_is_junction_node(o_from_terminal)
    b_from_load = get_terminal_has_load(o_from_terminal)
    b_from_dg = get_terminal_has_distributed_generation(o_from_terminal)

    b_to_busbar = get_terminal_is_busbar(o_to_terminal)
    b_to_junction = get_terminal_is_junction_node(o_to_terminal)
    b_to_load = get_terminal_has_load(o_to_terminal)
    b_to_dg = get_terminal_has_distributed_generation(o_to_terminal)

    if b_from_busbar and b_to_junction:
        return "busbar_to_junction"

    if b_from_busbar and b_to_busbar:
        return "busbar_to_busbar"

    if b_from_junction and b_to_busbar:
        return "junction_to_busbar"

    if b_from_junction and b_to_junction:
        return "junction_to_junction"

    if b_from_junction and b_to_dg:
        return "junction_to_distributed_generation"

    if b_from_busbar and b_to_dg:
        return "busbar_to_distributed_generation"

    if b_from_busbar and b_to_load:
        return "busbar_to_load"

    if b_from_dg and b_to_junction:
        return "distributed_generation_to_junction"

    if b_from_dg and b_to_busbar:
        return "distributed_generation_to_busbar"

    if b_from_load and b_to_busbar:
        return "load_to_busbar"

    return "other"


# =============================================================================
# Distributed Generation Detection Functions
# =============================================================================

def get_unique_distributed_generators_from_terminals(l_terminals):
    """
    Returns unique in-service distributed generators connected to given terminals.
    """

    l_distributed_generators = []
    l_seen_full_names = set()

    for o_terminal in l_terminals:
        for o_dg in get_terminal_connected_distributed_generators(o_terminal):
            s_full_name = _get_safe_full_name(o_dg)

            if s_full_name in l_seen_full_names:
                continue

            l_seen_full_names.add(s_full_name)
            l_distributed_generators.append(o_dg)

    return l_distributed_generators


def get_dg_summary_from_terminals(l_terminals):
    """
    Generic DG summary for a list of terminals.
    """

    l_dg = get_unique_distributed_generators_from_terminals(
        l_terminals,
    )

    l_dg_names = [
        _get_safe_name(o_dg)
        for o_dg in l_dg
    ]

    f_total_capacity_mva = round(
        sum([
            get_distributed_generator_capacity_mva(o_dg)
            for o_dg in l_dg
        ]),
        3,
    )

    return {
        "has_dg": get_boolean_value(len(l_dg) > 0),
        "count": len(l_dg),
        "capacity_mva": f_total_capacity_mva,
        "names": " -> ".join(l_dg_names),
        "objects": l_dg,
    }


def get_ordered_terminals_along_protected_corridor(d_corridor):
    """
    Returns terminals along the protected corridor in relay-to-remote direction.
    """

    l_corridor_terminals = []

    o_current_terminal = d_corridor.get("relay_busbar")
    l_line_sections = d_corridor.get("line_sections", [])

    if o_current_terminal is None:
        return l_corridor_terminals

    l_corridor_terminals.append(o_current_terminal)

    for o_line in l_line_sections:
        o_next_terminal = _get_opposite_terminal(
            o_line,
            o_current_terminal,
        )

        if o_next_terminal is None:
            break

        l_corridor_terminals.append(o_next_terminal)
        o_current_terminal = o_next_terminal

    return l_corridor_terminals


def summarize_relay_busbar_dg(d_corridor):
    """
    DG exactly at relay/source busbar.
    """

    o_relay_busbar = d_corridor.get("relay_busbar")

    if o_relay_busbar is None:
        return get_dg_summary_from_terminals([])

    return get_dg_summary_from_terminals(
        [o_relay_busbar],
    )


def summarize_subsequent_busbar_dg(d_corridor):
    """
    DG exactly at the subsequent busbar,
    i.e. the far end of the protected corridor.
    """

    o_subsequent_busbar = d_corridor.get("subsequent_busbar")

    if o_subsequent_busbar is None:
        return get_dg_summary_from_terminals([])

    return get_dg_summary_from_terminals(
        [o_subsequent_busbar],
    )


def summarize_protected_corridor_dg(d_corridor):
    """
    DG on intermediate junction terminals inside the protected corridor.

    Excludes:
        - relay/source busbar
        - remote/subsequent busbar
    """

    l_corridor_terminals = get_ordered_terminals_along_protected_corridor(
        d_corridor,
    )

    o_relay_busbar = d_corridor.get("relay_busbar")
    o_subsequent_busbar = d_corridor.get("subsequent_busbar")

    l_intermediate_terminals = []

    for o_terminal in l_corridor_terminals:
        if o_terminal == o_relay_busbar:
            continue

        if o_terminal == o_subsequent_busbar:
            continue

        l_intermediate_terminals.append(o_terminal)

    d_summary = get_dg_summary_from_terminals(
        l_intermediate_terminals,
    )

    d_summary["terminal_names"] = " -> ".join([
        _get_safe_name(o_terminal)
        for o_terminal in l_intermediate_terminals
    ])

    return d_summary


def summarize_downstream_branch_dg(d_shortest_branch):
    """
    DG on the immediate downstream branch after the remote busbar.

    Excludes:
        - branch start terminal, which is the remote busbar
        - branch end terminal, which is the far-end busbar

    Keeps:
        - intermediate junction terminals on the downstream branch
    """

    l_branch_terminals = d_shortest_branch.get("Branch Terminals", [])

    if len(l_branch_terminals) <= 2:
        return get_dg_summary_from_terminals([])

    l_intermediate_branch_terminals = l_branch_terminals[1:-1]

    d_summary = get_dg_summary_from_terminals(
        l_intermediate_branch_terminals,
    )

    d_summary["terminal_names"] = " -> ".join([
        _get_safe_name(o_terminal)
        for o_terminal in l_intermediate_branch_terminals
    ])

    return d_summary


def summarize_remote_busbar_dg(d_corridor, d_shortest_branch):
    """
    DG at the remote busbar beyond the immediate downstream branch.

    It is diagnostic only and is not automatically used for infeed correction.

    For the selected downstream branch:
        start terminal      = subsequent busbar of protected corridor
        intermediate nodes  = immediate downstream branch
        end terminal        = far-end (remote) busbar / next network boundary

    Therefore this function only checks the far-end busbar for now.
    It intentionally does NOT include intermediate branch DGs, because those are
    already counted as DownstreamBranchDG.
    """

    l_branch_terminals = d_shortest_branch.get("Branch Terminals", [])

    if len(l_branch_terminals) < 2:
        d_summary = get_dg_summary_from_terminals([])
        d_summary["terminal_names"] = ""
        return d_summary

    o_far_end_terminal = l_branch_terminals[-1]

    d_summary = get_dg_summary_from_terminals(
        [o_far_end_terminal],
    )

    d_summary["terminal_names"] = _get_safe_name(
        o_far_end_terminal,
    )

    return d_summary


def summarize_dg_by_corridor_location(d_corridor, d_shortest_branch):
    """
    Returns all directional DG context categories.
    """

    d_relay_busbar = summarize_relay_busbar_dg(
        d_corridor,
    )

    d_protected_corridor = summarize_protected_corridor_dg(
        d_corridor,
    )

    d_subsequent_busbar = summarize_subsequent_busbar_dg(
        d_corridor,
    )

    d_downstream_branch = summarize_downstream_branch_dg(
        d_shortest_branch,
    )

    d_remote_busbar = summarize_remote_busbar_dg(
        d_corridor,
        d_shortest_branch,
    )

    return {
        "relay_busbar": d_relay_busbar,
        "protected_corridor": d_protected_corridor,
        "subsequent_busbar": d_subsequent_busbar,
        "downstream_branch": d_downstream_branch,
        "remote_busbar": d_remote_busbar,
    }


# =============================================================================
# Network Extraction Functions
# =============================================================================

def _extract_line_data(o_grid):
    _print("\n⚙️ Extracting line data...")

    l_all_line_objects_raw = o_pf.GetCalcRelevantObjects("*.ElmLne") or []
    l_all_line_objects = []

    for o_line in l_all_line_objects_raw:
        if get_line_is_available_for_topology(
                o_line,
        ):
            l_all_line_objects.append(
                o_line,
            )
        else:
            if b_enable_debug and _is_object_in_service(
                    o_line,
            ):
                d_switch_state = get_line_switch_state_summary(
                    o_line,
                )

                _print(
                    f"⚠️ Skipping open-switched line: {_get_safe_name(o_line)} | "
                    f"bus1_switch_closed={d_switch_state.get('bus1_switch_closed')} | "
                    f"bus2_switch_closed={d_switch_state.get('bus2_switch_closed')}"
                )

    l_line_data = []
    l_skipped_lines = []

    for o_line in l_all_line_objects:
        try:
            o_cubicle_from = _get_attribute_object(o_line, "bus1")
            o_cubicle_to = _get_attribute_object(o_line, "bus2")

            if o_cubicle_from is None or o_cubicle_to is None:
                l_skipped_lines.append((_get_safe_name(o_line), "missing bus1/bus2 cubicle"))
                continue

            o_terminal_from = _get_terminal_from_cubicle(o_cubicle_from)
            o_terminal_to = _get_terminal_from_cubicle(o_cubicle_to)

            if o_terminal_from is None or o_terminal_to is None:
                l_skipped_lines.append((_get_safe_name(o_line), "missing terminal cterm"))
                continue

            if (
                    not _is_object_inside_grid(o_terminal_from, o_grid)
                    or not _is_object_inside_grid(o_terminal_to, o_grid)
            ):
                l_skipped_lines.append((_get_safe_name(o_line), "outside selected grid"))
                continue

            s_from_bus_name = _get_safe_name(o_terminal_from)
            s_to_bus_name = _get_safe_name(o_terminal_to)

            if s_from_bus_name == s_to_bus_name:
                l_skipped_lines.append((_get_safe_name(o_line), "same from/to bus name"))
                continue

            f_line_length_km = _get_attribute_float(o_line, "dline", 0.0)
            f_line_r_ohm, f_line_x_ohm = _get_line_impedance(o_line)

            c_line_z_ohm = complex(f_line_r_ohm, f_line_x_ohm)

            if abs(c_line_z_ohm) > f_zero_tolerance:
                c_line_y_siemens = 1.0 / c_line_z_ohm
            else:
                c_line_y_siemens = complex(0.0, 0.0)

            l_line_data.append({
                "from_bus_name": s_from_bus_name,
                "from_cubicle_name": _get_safe_name(o_cubicle_from),
                "line_name": _get_safe_name(o_line),
                "to_cubicle_name": _get_safe_name(o_cubicle_to),
                "to_bus_name": s_to_bus_name,
                "length_km": f_line_length_km,
                "R (ohm)": f_line_r_ohm,
                "X (ohm)": f_line_x_ohm,
                "Z (ohm)": f"{c_line_z_ohm.real:.3f}{c_line_z_ohm.imag:+.3f}j",
                "G (siemens)": c_line_y_siemens.real,
                "B (siemens)": c_line_y_siemens.imag,
                "Y (siemens)": f"{c_line_y_siemens.real:.3f}{c_line_y_siemens.imag:+.3f}j",
                "_R_value": f_line_r_ohm,
                "_X_value": f_line_x_ohm,
                "_G_value": c_line_y_siemens.real,
                "_B_value": c_line_y_siemens.imag,
                "_pf_line_object": o_line,
                "_pf_terminal_from": o_terminal_from,
                "_pf_terminal_to": o_terminal_to,
                "_pf_cubicle_from": o_cubicle_from,
                "_pf_cubicle_to": o_cubicle_to,
            })

        except Exception as o_error:
            l_skipped_lines.append((_get_safe_name(o_line), str(o_error)))

    _print(f"✅ Collected {len(l_line_data)} valid lines inside selected grid.")

    if l_skipped_lines:
        _print(f"⚠️ Skipped lines: {len(l_skipped_lines)}")

    return l_line_data, l_skipped_lines


def extract_network_once(o_grid):
    """
    Extracts all required network data once.
    """

    l_line_data, l_skipped_lines = _extract_line_data(
        o_grid,
    )

    return {
        "line_data": l_line_data,
        "skipped_lines": l_skipped_lines,
    }


# =============================================================================
# Protected Busbar-to-Busbar Corridor Functions
# =============================================================================

def join_line_names(l_line_objects):
    return "; ".join([
        _get_safe_name(o_line)
        for o_line in l_line_objects
    ])


def calculate_line_path_impedance(l_line_sections):
    f_total_r = 0.0
    f_total_x = 0.0
    f_total_length = 0.0

    for o_line in l_line_sections:
        f_r, f_x = _get_line_impedance(o_line)
        f_length = get_line_length_value(o_line)

        f_total_r += f_r
        f_total_x += f_x
        f_total_length += f_length

    return {
        "total_r_ohm": round(f_total_r, 3),
        "total_x_ohm": round(f_total_x, 3),
        "total_length_km": round(f_total_length, 3),
        "hop_count": len(l_line_sections),
    }


def get_next_line_from_junction(o_junction_terminal, o_previous_line, l_visited_lines):
    for o_element in get_terminal_connected_elements(o_junction_terminal):
        if _get_safe_class_name(o_element) != "ElmLne":
            continue

        if not _is_object_in_service(o_element):
            continue

        if o_element == o_previous_line:
            continue

        if o_element in l_visited_lines:
            continue

        return o_element

    return None


def trace_protected_corridor_from_relay_busbar(o_relay_busbar, o_first_line):
    if not get_terminal_is_busbar(o_relay_busbar):
        return None

    l_line_sections = []
    l_junction_nodes = []
    l_visited_lines = []

    o_current_terminal = o_relay_busbar
    o_current_line = o_first_line

    while o_current_line is not None:
        if o_current_line in l_visited_lines:
            break

        l_visited_lines.append(o_current_line)
        l_line_sections.append(o_current_line)

        o_next_terminal = _get_opposite_terminal(
            o_current_line,
            o_current_terminal,
        )

        if o_next_terminal is None:
            break

        if get_terminal_is_busbar(o_next_terminal):
            d_impedance = calculate_line_path_impedance(l_line_sections)

            return {
                "relay_busbar": o_relay_busbar,
                "subsequent_busbar": o_next_terminal,
                "line_sections": l_line_sections,
                "junction_nodes": l_junction_nodes,
                "first_line_section": l_line_sections[0],
                "last_line_section": l_line_sections[-1],
                "protected_corridor_id": join_line_names(l_line_sections),
                "total_r_ohm": d_impedance["total_r_ohm"],
                "total_x_ohm": d_impedance["total_x_ohm"],
                "total_length_km": d_impedance["total_length_km"],
                "hop_count": d_impedance["hop_count"],
            }

        if get_terminal_is_junction_node(o_next_terminal):
            l_junction_nodes.append(o_next_terminal)

            o_next_line = get_next_line_from_junction(
                o_next_terminal,
                o_current_line,
                l_visited_lines,
            )

            o_current_terminal = o_next_terminal
            o_current_line = o_next_line

            continue

        break

    return None


def build_all_directional_protected_corridors(l_line_data):
    l_protected_corridors = []
    l_seen_corridors = set()

    for d_line in l_line_data:
        o_line = d_line.get("_pf_line_object")
        o_terminal_from = d_line.get("_pf_terminal_from")
        o_terminal_to = d_line.get("_pf_terminal_to")

        if o_line is None or o_terminal_from is None or o_terminal_to is None:
            continue

        for o_relay_busbar in [o_terminal_from, o_terminal_to]:
            if not get_terminal_is_busbar(o_relay_busbar):
                continue

            d_corridor = trace_protected_corridor_from_relay_busbar(
                o_relay_busbar,
                o_line,
            )

            if d_corridor is None:
                continue

            s_key = (
                _get_safe_full_name(d_corridor["relay_busbar"]),
                _get_safe_full_name(d_corridor["subsequent_busbar"]),
                d_corridor["protected_corridor_id"],
            )

            if s_key in l_seen_corridors:
                continue

            o_relay_cubicle = get_cubicle_for_line_at_terminal(
                d_corridor["first_line_section"],
                d_corridor["relay_busbar"],
            )

            d_corridor["relay_cubicle"] = o_relay_cubicle

            l_seen_corridors.add(s_key)
            l_protected_corridors.append(d_corridor)

    _print(f"✅ Protected busbar-to-busbar corridors created: {len(l_protected_corridors)}")

    return l_protected_corridors


def detect_protected_corridors_once(d_network):
    """
    Detects all protected busbar-to-busbar corridors once.
    """

    l_line_data = d_network.get(
        "line_data",
        [],
    )

    l_protected_corridors = build_all_directional_protected_corridors(
        l_line_data,
    )

    return l_protected_corridors


def trace_branch_from_busbar_to_next_busbar(o_start_busbar, o_first_line, l_excluded_lines):
    l_branch_lines = []
    l_branch_terminals = [o_start_busbar]
    l_visited_lines = l_excluded_lines[:]

    o_current_terminal = o_start_busbar
    o_current_line = o_first_line

    while o_current_line is not None:
        if o_current_line in l_visited_lines:
            break

        l_visited_lines.append(o_current_line)
        l_branch_lines.append(o_current_line)

        o_next_terminal = _get_opposite_terminal(
            o_current_line,
            o_current_terminal,
        )

        if o_next_terminal is None:
            break

        l_branch_terminals.append(o_next_terminal)

        if get_terminal_is_busbar(o_next_terminal):
            return build_branch_summary(
                l_branch_lines,
                l_branch_terminals,
            )

        if get_terminal_is_junction_node(o_next_terminal):
            o_next_line = get_next_line_from_junction(
                o_next_terminal,
                o_current_line,
                l_visited_lines,
            )

            o_current_terminal = o_next_terminal
            o_current_line = o_next_line

            continue

        break

    return None


def line_connects_terminal_pair_by_full_name(
        o_line,
        s_terminal_a_full_name,
        s_terminal_b_full_name,
):
    """
    Returns True if a line directly connects the two given terminals.

    This is used to reject physical parallel protected-corridor return paths.

    Example:
        Protected corridor: Terminal_HV_A -> Terminal_HV_B via Line_HV_A-B-1

        Candidate from Terminal_HV_B:
            Line_HV_A-B-2

        Since Line_HV_A-B-2 directly connects Terminal_HV_A and Terminal_HV_B,
        it is not a downstream branch. It is a parallel return path.
    """

    o_terminal_1, o_terminal_2, _, _ = get_line_terminals(
        o_line,
    )

    s_terminal_1_full_name = _get_safe_full_name(
        o_terminal_1,
    )

    s_terminal_2_full_name = _get_safe_full_name(
        o_terminal_2,
    )

    return {s_terminal_1_full_name, s_terminal_2_full_name} == {s_terminal_a_full_name, s_terminal_b_full_name}


def build_empty_branch_summary(o_subsequent_busbar):
    """
    Returns an empty downstream branch dictionary.
    """

    return {
        "Branch Lines": [],
        "Branch Terminals": [o_subsequent_busbar] if o_subsequent_busbar is not None else [],

        "Branch ID": "",
        "Remote Node ID": "",
        "Hop Count": 0,

        "Branch Length": 0.0,
        "Branch Resistance": 0.0,
        "Branch Reactance": 0.0,

        "Has Parallel": False,
        "Parallel Count": 0,
        "Parallel Lines": "",

        "Junction Node Count": 0,
        "Distributed Generation Count": 0,

        "Contains Load": False,
        "Contains Distributed Generation": False,
        "Total Load P MW": 0.0,
        "Distributed Generators": [],
    }


def get_branch_far_end_busbar(d_branch):
    """
    Returns the far-end busbar of a traced downstream branch.
    """

    if not d_branch:
        return None

    l_branch_terminals = d_branch.get(
        "Branch Terminals",
        [],
    )

    if not l_branch_terminals:
        return None

    return l_branch_terminals[-1]


def filter_valid_downstream_branches_excluding_relay_return(
        d_corridor,
        l_all_corridors=None,
):
    """
    Returns valid downstream branches from the subsequent busbar.

    Excludes:
        - current protected corridor line sections
        - physical parallel alternatives to the protected corridor
        - branches that directly connect subsequent_busbar back to relay_busbar
        - branches whose far end is relay_busbar
        - branches whose far end is subsequent_busbar
    """

    o_relay_busbar = d_corridor.get(
        "relay_busbar",
    )

    o_subsequent_busbar = d_corridor.get(
        "subsequent_busbar",
    )

    l_protected_corridor_lines = d_corridor.get(
        "line_sections",
        [],
    )

    if o_relay_busbar is None or o_subsequent_busbar is None:
        return []

    s_relay_busbar_full_name = _get_safe_full_name(
        o_relay_busbar,
    )

    s_subsequent_busbar_full_name = _get_safe_full_name(
        o_subsequent_busbar,
    )

    l_total_exclusion_lines = l_protected_corridor_lines[:]

    if l_all_corridors is not None:
        l_parallel_corridors = find_parallel_protected_corridors(
            d_corridor,
            l_all_corridors,
        )

        for d_parallel_corridor in l_parallel_corridors:
            for o_line in d_parallel_corridor.get("line_sections", []):
                if o_line not in l_total_exclusion_lines:
                    l_total_exclusion_lines.append(
                        o_line,
                    )

    l_valid_branches = []

    for o_candidate_line in get_terminal_connected_elements(
            o_subsequent_busbar,
    ):
        if _get_safe_class_name(o_candidate_line) != "ElmLne":
            continue

        if not _is_object_in_service(o_candidate_line):
            continue

        if o_candidate_line in l_total_exclusion_lines:
            continue

        # ---------------------------------------------------------------------
        # Critical direct relay-return rejection.
        #
        # This rejects:
        #   A -> B via Line_HV_A-B-1 selecting Line_HV_A-B-2
        #   A -> B via Line_HV_A-B-2 selecting Line_HV_A-B-1
        # ---------------------------------------------------------------------
        if line_connects_terminal_pair_by_full_name(
                o_candidate_line,
                s_relay_busbar_full_name,
                s_subsequent_busbar_full_name,
        ):
            continue

        d_branch = trace_branch_from_busbar_to_next_busbar(
            o_subsequent_busbar,
            o_candidate_line,
            l_total_exclusion_lines[:],
        )

        if d_branch is None:
            continue

        o_far_end_busbar = get_branch_far_end_busbar(
            d_branch,
        )

        if o_far_end_busbar is None:
            continue

        s_far_end_busbar_full_name = _get_safe_full_name(
            o_far_end_busbar,
        )

        if s_far_end_busbar_full_name == s_relay_busbar_full_name:
            continue

        if s_far_end_busbar_full_name == s_subsequent_busbar_full_name:
            continue

        l_valid_branches.append(
            d_branch,
        )

    return l_valid_branches


def select_zone2_downstream_branch_group(d_corridor, l_all_corridors=None):
    """
    Selects the correct downstream branch group for Zone 2.
    """
    o_subsequent_busbar = d_corridor.get("subsequent_busbar")

    # Pass the context list forward to the filter
    l_valid_branches = filter_valid_downstream_branches_excluding_relay_return(
        d_corridor,
        l_all_corridors=l_all_corridors
    )

    if not l_valid_branches:
        return {
            "selected_branch": build_empty_branch_summary(o_subsequent_busbar),
            "selected_parallel_branches": [],
            "selected_remote_busbar": None,
            "selected_remote_busbar_name": "",
            "zone2_branch_selection_method": "no_valid_downstream_branch",
            "zone2_reach_calculation_method": "simple_no_downstream_branch",
            "zone2_selected_branch_has_parallel": 0,
            "zone2_selected_branch_parallel_count": 0,
        }

    d_remote_busbar_to_branches = defaultdict(list)
    for d_branch in l_valid_branches:
        o_remote_busbar = get_branch_far_end_busbar(d_branch)
        s_remote_busbar_key = _get_safe_full_name(o_remote_busbar)
        d_remote_busbar_to_branches[s_remote_busbar_key].append(d_branch)

    l_remote_group_summaries = []
    for s_remote_busbar_key, l_branches in d_remote_busbar_to_branches.items():
        l_sorted_branches = sorted(
            l_branches,
            key=lambda d: (
                float(d.get("Branch Reactance", 0.0) or 0.0),
                float(d.get("Branch Length", 0.0) or 0.0),
                float(d.get("Branch Resistance", 0.0) or 0.0),
            ),
        )
        d_shortest_branch_to_remote = l_sorted_branches[0]

        l_remote_group_summaries.append({
            "remote_busbar_key": s_remote_busbar_key,
            "selected_branch": d_shortest_branch_to_remote,
            "parallel_branches": l_sorted_branches,
            "parallel_count": len(l_sorted_branches),
            "remote_busbar": get_branch_far_end_busbar(d_shortest_branch_to_remote),
        })

    l_remote_group_summaries = sorted(
        l_remote_group_summaries,
        key=lambda d: (
            float(d["selected_branch"].get("Branch Reactance", 0.0) or 0.0),
            float(d["selected_branch"].get("Branch Length", 0.0) or 0.0),
            float(d["selected_branch"].get("Branch Resistance", 0.0) or 0.0),
        ),
    )

    d_selected_group = l_remote_group_summaries[0]
    i_parallel_count = int(d_selected_group["parallel_count"])
    b_has_parallel = i_parallel_count > 1

    if b_has_parallel:
        s_zone2_reach_calculation_method = "complex_parallel_zone2_reach"
        s_zone2_branch_selection_method = "shortest_valid_remote_busbar_group_with_parallel"
    else:
        s_zone2_reach_calculation_method = "simple_zone2_reach"
        s_zone2_branch_selection_method = "shortest_valid_downstream_branch"

    return {
        "selected_branch": d_selected_group["selected_branch"],
        "selected_parallel_branches": d_selected_group["parallel_branches"],
        "selected_remote_busbar": d_selected_group["remote_busbar"],
        "selected_remote_busbar_name": _get_safe_name(d_selected_group["remote_busbar"]),
        "zone2_branch_selection_method": s_zone2_branch_selection_method,
        "zone2_reach_calculation_method": s_zone2_reach_calculation_method,
        "zone2_selected_branch_has_parallel": get_boolean_value(b_has_parallel),
        "zone2_selected_branch_parallel_count": i_parallel_count,
    }


def select_parallel_branch_for_complex_zone2(
        d_selected_branch,
        l_parallel_branches,
):
    """
    Selects the parallel branch to be used as Z3 in the complex Zone 2 formula.

    The selected shortest branch is Z2.
    The other parallel branch is Z3.

    If more than one alternative exists, choose the shortest alternative by
    reactance, length, then resistance.
    """

    l_candidate_parallel_branches = []

    for d_branch in l_parallel_branches:
        if d_branch is d_selected_branch:
            continue

        s_branch_id = join_line_names(
            d_branch.get("Branch Lines", [])
        )

        s_selected_branch_id = join_line_names(
            d_selected_branch.get("Branch Lines", [])
        )

        if s_branch_id == s_selected_branch_id:
            continue

        l_candidate_parallel_branches.append(
            d_branch,
        )

    if not l_candidate_parallel_branches:
        return None

    l_candidate_parallel_branches = sorted(
        l_candidate_parallel_branches,
        key=lambda d: (
            float(d.get("Branch Reactance", 0.0) or 0.0),
            float(d.get("Branch Length", 0.0) or 0.0),
            float(d.get("Branch Resistance", 0.0) or 0.0),
        ),
    )

    return l_candidate_parallel_branches[0]


def select_zone3_longest_valid_downstream_branch(d_corridor, l_all_corridors=None):
    """
    Selects the longest valid downstream branch for Zone 3.
    """
    o_subsequent_busbar = d_corridor.get("subsequent_busbar")

    # Pass the context list forward to filter out co-linear protected corridor duplicates
    l_valid_branches = filter_valid_downstream_branches_excluding_relay_return(
        d_corridor,
        l_all_corridors=l_all_corridors
    )

    if not l_valid_branches:
        return build_empty_branch_summary(o_subsequent_busbar)

    # Sort in reverse to pull the longest downstream branch by Length/Reactance
    l_valid_branches = sorted(
        l_valid_branches,
        key=lambda d: (
            float(d.get("Branch Length", 0.0) or 0.0),
            float(d.get("Branch Reactance", 0.0) or 0.0),
            float(d.get("Branch Resistance", 0.0) or 0.0),
        ),
        reverse=True,
    )

    return l_valid_branches[0]


def count_forward_parallel_branch_groups_for_corridor(d_corridor, l_all_corridors=None):
    """
    Counts forward valid downstream parallel branch groups for a corridor.
    """
    l_valid_branches = filter_valid_downstream_branches_excluding_relay_return(
        d_corridor,
        l_all_corridors=l_all_corridors
    )

    d_end_busbar_to_branches = defaultdict(list)
    for d_branch in l_valid_branches:
        o_end_busbar = get_branch_far_end_busbar(d_branch)
        if o_end_busbar is None:
            continue
        d_end_busbar_to_branches[_get_safe_full_name(o_end_busbar)].append(d_branch)

    i_parallel_group_count = 0
    i_max_parallel_branch_count = 0

    for _, l_branches in d_end_busbar_to_branches.items():
        if len(l_branches) > 1:
            i_parallel_group_count += 1
            i_max_parallel_branch_count = max(i_max_parallel_branch_count, len(l_branches))

    return i_parallel_group_count, i_max_parallel_branch_count



def line_connects_terminal_pair_by_full_name(
        o_line,
        s_terminal_a_full_name,
        s_terminal_b_full_name,
):
    """
    Returns True if a line directly connects the two given terminals.
    """

    if o_line is None:
        return False

    o_terminal_1, o_terminal_2, _, _ = get_line_terminals(
        o_line,
    )

    if o_terminal_1 is None or o_terminal_2 is None:
        return False

    s_terminal_1_full_name = _get_safe_full_name(
        o_terminal_1,
    )

    s_terminal_2_full_name = _get_safe_full_name(
        o_terminal_2,
    )

    return {
        s_terminal_1_full_name,
        s_terminal_2_full_name,
    } == {
        s_terminal_a_full_name,
        s_terminal_b_full_name,
    }


def get_parallel_lines_for_line(o_line):
    """
    Returns physical parallel lines directly connecting the same two terminals.
    """

    if o_line is None:
        return []

    o_terminal_1, o_terminal_2, _, _ = get_line_terminals(
        o_line,
    )

    if o_terminal_1 is None or o_terminal_2 is None:
        return []

    s_terminal_1_full_name = _get_safe_full_name(
        o_terminal_1,
    )

    s_terminal_2_full_name = _get_safe_full_name(
        o_terminal_2,
    )

    l_parallel_lines = []

    for o_candidate_line in get_terminal_connected_lines(
            o_terminal_1,
    ):
        if o_candidate_line == o_line:
            continue

        if line_connects_terminal_pair_by_full_name(
                o_candidate_line,
                s_terminal_1_full_name,
                s_terminal_2_full_name,
        ):
            l_parallel_lines.append(
                o_candidate_line,
            )

    return get_unique_objects(
        l_parallel_lines,
    )


def get_parallel_lines_for_line_list(l_lines):
    """
    Returns all physical parallel lines for a list of line sections.
    """

    l_parallel_lines = []

    for o_line in l_lines:
        l_parallel_lines.extend(
            get_parallel_lines_for_line(
                o_line,
            )
        )

    return get_unique_objects(
        l_parallel_lines,
    )


def build_branch_summary(
        l_branch_lines,
        l_branch_terminals,
):
    """
    Builds one downstream branch summary.

    This function is used for both:
        - shortest downstream branch features
        - longest downstream branch features
    """

    d_impedance = calculate_line_path_impedance(
        l_branch_lines,
    )

    l_parallel_lines = get_parallel_lines_for_line_list(
        l_branch_lines,
    )

    l_distributed_generators = get_unique_distributed_generators_from_terminals(
        l_branch_terminals,
    )

    l_junction_nodes = [
        o_terminal
        for o_terminal in l_branch_terminals
        if get_terminal_is_junction_node(
            o_terminal,
        )
    ]

    b_contains_load = any([
        get_terminal_has_load(
            o_terminal,
        )
        for o_terminal in l_branch_terminals
    ])

    f_total_load_p_mw = sum([
        get_terminal_load_p_mw(
            o_terminal,
        )
        for o_terminal in l_branch_terminals
    ])

    return {
        "Branch Lines": l_branch_lines,
        "Branch Terminals": l_branch_terminals,

        # New branch identity fields
        "Branch ID": join_line_names(
            l_branch_lines,
        ),
        "Remote Node ID": _get_safe_name(
            l_branch_terminals[-1],
        ) if l_branch_terminals else "",
        "Hop Count": len(
            l_branch_lines,
        ),

        # Existing electrical fields
        "Branch Length": d_impedance.get(
            "total_length_km",
            0.0,
        ),
        "Branch Resistance": d_impedance.get(
            "total_r_ohm",
            0.0,
        ),
        "Branch Reactance": d_impedance.get(
            "total_x_ohm",
            0.0,
        ),

        # Parallel branch fields
        "Has Parallel": len(
            l_parallel_lines,
        ) > 0,
        "Parallel Count": len(
            l_parallel_lines,
        ),
        "Parallel Lines": join_line_names(
            l_parallel_lines,
        ),

        # New topology fields
        "Junction Node Count": len(
            get_unique_objects(
                l_junction_nodes,
            )
        ),
        "Distributed Generation Count": len(
            l_distributed_generators,
        ),

        # Keep old/internal fields so existing code does not break
        "Contains Load": b_contains_load,
        "Contains Distributed Generation": len(
            l_distributed_generators,
        ) > 0,
        "Distributed Generators": l_distributed_generators,
        "Total Load P MW": round(
            f_total_load_p_mw,
            3,
        ),
    }


# =============================================================================
# Parallel Corridor Detection Functions
# =============================================================================

def protected_corridors_have_same_busbar_pair(d_corridor_a, d_corridor_b):
    """
    Checks whether two protected corridors connect the same two busbars,
    regardless of direction.
    """

    o_a_1 = d_corridor_a.get("relay_busbar")
    o_a_2 = d_corridor_a.get("subsequent_busbar")
    o_b_1 = d_corridor_b.get("relay_busbar")
    o_b_2 = d_corridor_b.get("subsequent_busbar")

    if None in [o_a_1, o_a_2, o_b_1, o_b_2]:
        return False

    o_pair_a = frozenset([
        _get_safe_full_name(o_a_1),
        _get_safe_full_name(o_a_2),
    ])

    o_pair_b = frozenset([
        _get_safe_full_name(o_b_1),
        _get_safe_full_name(o_b_2),
    ])

    return o_pair_a == o_pair_b


def get_protected_corridor_line_full_name_set(d_corridor):
    """
    Returns a direction-independent set of physical line full names.

    This prevents the same segmented corridor in reverse direction from being
    detected as a parallel corridor.
    """

    return frozenset([
        _get_safe_full_name(o_line)
        for o_line in d_corridor.get("line_sections", [])
    ])


def find_parallel_protected_corridors(d_target_corridor, l_all_corridors):
    """
    Finds true parallel corridors that connect the same two busbars.

    Important:
    - Same busbar pair is required.
    - Same physical line-section set is excluded.
      This prevents C-D forward and C-D reverse from being treated as parallel.
    """

    l_parallel_corridors = []

    o_target_line_set = get_protected_corridor_line_full_name_set(
        d_target_corridor,
    )

    for d_other_corridor in l_all_corridors:
        if d_other_corridor is d_target_corridor:
            continue

        if not protected_corridors_have_same_busbar_pair(
                d_target_corridor,
                d_other_corridor,
        ):
            continue

        o_other_line_set = get_protected_corridor_line_full_name_set(
            d_other_corridor,
        )

        # Same physical corridor, possibly reverse direction.
        # Do not count as parallel.
        if o_other_line_set == o_target_line_set:
            continue

        l_parallel_corridors.append(d_other_corridor)

    return l_parallel_corridors


def summarize_parallel_protected_corridors(d_target_corridor, l_all_corridors):
    """
    Returns parallel corridor summary for the target corridor.
    """

    l_parallel_corridors = find_parallel_protected_corridors(
        d_target_corridor,
        l_all_corridors,
    )

    l_parallel_line_names = []
    f_parallel_r_ohm = 0.0
    f_parallel_x_ohm = 0.0
    f_parallel_length_km = 0.0

    if l_parallel_corridors:
        # For now, use the first parallel alternative.
        # Later we can extend this if there are more than two parallel corridors.
        d_first_parallel = l_parallel_corridors[0]

        l_parallel_line_names = [
            _get_safe_name(o_line)
            for o_line in d_first_parallel.get("line_sections", [])
        ]

        f_parallel_r_ohm = d_first_parallel.get("total_r_ohm", 0.0)
        f_parallel_x_ohm = d_first_parallel.get("total_x_ohm", 0.0)
        f_parallel_length_km = d_first_parallel.get("total_length_km", 0.0)

    return {
        "is_parallel": len(l_parallel_corridors) > 0,
        "parallel_count": len(l_parallel_corridors) + 1 if l_parallel_corridors else 1,
        "parallel_alternative_lines": "; ".join(l_parallel_line_names),
        "parallel_r_ohm": f_parallel_r_ohm,
        "parallel_x_ohm": f_parallel_x_ohm,
        "parallel_length_km": f_parallel_length_km,
        "parallel_corridors": l_parallel_corridors,
    }


# =============================================================================
# Next Node / Parallel Summary Functions
# =============================================================================

def get_next_node_summary(o_next_terminal, l_excluded_lines):
    d_summary = {
        "next_node_line_count": 0,
        "next_node_busbar_count": 0,
        "next_node_junction_node_count": 0,
        "next_node_load_count": 0,
        "next_node_generation_count": 0,
        "next_node_distributed_generation_count": 0,
        "next_node_parallel_line_count": 0,
    }

    for o_element in get_terminal_connected_elements(o_next_terminal):
        s_class_name = _get_safe_class_name(o_element)

        if s_class_name == "ElmLne" and _is_object_in_service(o_element):
            if o_element in l_excluded_lines:
                continue

            d_summary["next_node_line_count"] += 1

            o_other_terminal = _get_opposite_terminal(o_element, o_next_terminal)

            if o_other_terminal is not None:
                if get_terminal_is_busbar(o_other_terminal):
                    d_summary["next_node_busbar_count"] += 1

                if get_terminal_is_junction_node(o_other_terminal):
                    d_summary["next_node_junction_node_count"] += 1

        elif s_class_name in ["ElmLod", "ElmLodlv"] and _is_object_in_service(o_element):
            d_summary["next_node_load_count"] += 1

        elif s_class_name in ["ElmSym", "ElmGenstat", "ElmPvsys"] and _is_object_in_service(o_element):
            d_summary["next_node_generation_count"] += 1

            if s_class_name in ["ElmGenstat", "ElmPvsys"]:
                d_summary["next_node_distributed_generation_count"] += 1

    return d_summary


# =============================================================================
# Zone Formula Functions
# =============================================================================

def calculate_simple_zone2_reach_formula(
        f_connected_line_r_ohm,
        f_connected_line_x_ohm,
        f_branch_line_r_ohm,
        f_branch_line_x_ohm,
):
    """
    Simple Zone 2 reach formula.

    Z2_simple = grading_factor * (Z_connected + grading_factor * Z_branch)
    """

    f_zone2_r_ohm = round(
        i_reach_gf * (
                f_connected_line_r_ohm
                + i_reach_gf * f_branch_line_r_ohm
        ),
        3,
    )

    f_zone2_x_ohm = round(
        i_reach_gf * (
                f_connected_line_x_ohm
                + i_reach_gf * f_branch_line_x_ohm
        ),
        3,
    )

    return f_zone2_r_ohm, f_zone2_x_ohm


def calculate_complex_zone2_reach_formula(
        f_connected_line_r_ohm,
        f_connected_line_x_ohm,
        f_shortest_branch_r_ohm,
        f_shortest_branch_x_ohm,
        f_parallel_branch_r_ohm,
        f_parallel_branch_x_ohm,
):
    """
    Complex parallel Zone 2 reach formula.

    Z1 = protected corridor impedance
    Z2 = selected shortest downstream branch impedance
    Z3 = parallel downstream branch impedance

    Z2_complex = grading_factor * (
        Z1 + ((Z3 + (1 - grading_factor) * Z2) * grading_factor * Z2) / (Z2 + Z3)
    )
    """

    c_z1 = complex(
        f_connected_line_r_ohm,
        f_connected_line_x_ohm,
    )

    c_z2 = complex(
        f_shortest_branch_r_ohm,
        f_shortest_branch_x_ohm,
    )

    c_z3 = complex(
        f_parallel_branch_r_ohm,
        f_parallel_branch_x_ohm,
    )

    if abs(c_z2 + c_z3) <= f_zero_tolerance:
        return calculate_simple_zone2_reach_formula(
            f_connected_line_r_ohm,
            f_connected_line_x_ohm,
            f_shortest_branch_r_ohm,
            f_shortest_branch_x_ohm,
        )

    c_zone2_reach = i_reach_gf * (
            c_z1
            + (
                    (
                            c_z3
                            + (1.0 - i_reach_gf) * c_z2
                    )
                    * i_reach_gf
                    * c_z2
            )
            / (c_z2 + c_z3)
    )

    return round(c_zone2_reach.real, 3), round(c_zone2_reach.imag, 3)


def calculate_intended_zone2_reach_formula(
        f_connected_line_r_ohm,
        f_connected_line_x_ohm,
        f_branch_line_r_ohm,
        f_branch_line_x_ohm,
):
    """
    Intended Zone 2 reach formula.

    This mirrors the simple Zone 2 formula.
    """

    return calculate_simple_zone2_reach_formula(
        f_connected_line_r_ohm,
        f_connected_line_x_ohm,
        f_branch_line_r_ohm,
        f_branch_line_x_ohm,
    )


# =============================================================================
# Zone Calculation Functions
# =============================================================================

def calculate_distance_zone_reaches_for_corridor(d_corridor, d_downstream_branch, d_parallel_summary, l_all_corridors=None):
    """
    Calculates Zone 1, Zone 2, and Zone 3 reach values for one protected corridor.
    """
    f_protected_corridor_r = float(d_corridor.get("total_r_ohm", 0.0) or 0.0)
    f_protected_corridor_x = float(d_corridor.get("total_x_ohm", 0.0) or 0.0)
    f_protected_corridor_length = float(d_corridor.get("total_length_km", 0.0) or 0.0)

    # -------------------------------------------------------------------------
    # Zone 1
    # -------------------------------------------------------------------------
    f_zone1_r_reach = round(i_reach_gf * f_protected_corridor_r, 3)
    f_zone1_x_reach = round(i_reach_gf * f_protected_corridor_x, 3)

    # -------------------------------------------------------------------------
    # Zone 2 branch selection
    # -------------------------------------------------------------------------
    d_zone2_selection = select_zone2_downstream_branch_group(
        d_corridor,
        l_all_corridors=l_all_corridors
    )

    d_zone2_branch = d_zone2_selection.get("selected_branch", build_empty_branch_summary(d_corridor.get("subsequent_busbar")))
    l_zone2_parallel_branches = d_zone2_selection.get("selected_parallel_branches", [])

    f_zone2_selected_branch_r = float(d_zone2_branch.get("Branch Resistance", 0.0) or 0.0)
    f_zone2_selected_branch_x = float(d_zone2_branch.get("Branch Reactance", 0.0) or 0.0)
    f_zone2_selected_branch_length = float(d_zone2_branch.get("Branch Length", 0.0) or 0.0)
    s_zone2_branch_id = join_line_names(d_zone2_branch.get("Branch Lines", []))
    s_zone2_reach_calculation_method = d_zone2_selection.get("zone2_reach_calculation_method", "simple_zone2_reach")
    i_zone2_parallel_count = int(d_zone2_selection.get("zone2_selected_branch_parallel_count", 0) or 0)
    b_zone2_has_parallel = bool(d_zone2_selection.get("zone2_selected_branch_has_parallel", 0))

    # Intended Zone 2 Reach
    f_intended_zone2_r_reach, f_intended_zone2_x_reach = calculate_intended_zone2_reach_formula(
        f_protected_corridor_r, f_protected_corridor_x, f_zone2_selected_branch_r, f_zone2_selected_branch_x
    )

    # Actual Zone 2 Reach
    d_zone2_parallel_branch_for_complex = None
    f_zone2_parallel_branch_r = 0.0
    f_zone2_parallel_branch_x = 0.0
    f_zone2_parallel_branch_length = 0.0
    s_zone2_parallel_branch_id = ""

    if b_zone2_has_parallel and len(l_zone2_parallel_branches) > 1:
        d_zone2_parallel_branch_for_complex = select_parallel_branch_for_complex_zone2(
            d_zone2_branch, l_zone2_parallel_branches
        )
        if d_zone2_parallel_branch_for_complex is not None:
            f_zone2_parallel_branch_r = float(d_zone2_parallel_branch_for_complex.get("Branch Resistance", 0.0) or 0.0)
            f_zone2_parallel_branch_x = float(d_zone2_parallel_branch_for_complex.get("Branch Reactance", 0.0) or 0.0)
            f_zone2_parallel_branch_length = float(d_zone2_parallel_branch_for_complex.get("Branch Length", 0.0) or 0.0)
            s_zone2_parallel_branch_id = join_line_names(d_zone2_parallel_branch_for_complex.get("Branch Lines", []))

            f_zone2_r_reach, f_zone2_x_reach = calculate_complex_zone2_reach_formula(
                f_protected_corridor_r, f_protected_corridor_x,
                f_zone2_selected_branch_r, f_zone2_selected_branch_x,
                f_zone2_parallel_branch_r, f_zone2_parallel_branch_x
            )
            s_zone2_impedance_basis = "complex_parallel_shortest_branch_and_parallel_branch_impedance"
        else:
            f_zone2_r_reach, f_zone2_x_reach = calculate_simple_zone2_reach_formula(
                f_protected_corridor_r, f_protected_corridor_x, f_zone2_selected_branch_r, f_zone2_selected_branch_x
            )
            s_zone2_reach_calculation_method = "simple_zone2_reach"
            s_zone2_impedance_basis = "simple_selected_downstream_branch_impedance_no_parallel_alternative_found"
    else:
        f_zone2_r_reach, f_zone2_x_reach = calculate_simple_zone2_reach_formula(
            f_protected_corridor_r, f_protected_corridor_x, f_zone2_selected_branch_r, f_zone2_selected_branch_x
        )
        s_zone2_reach_calculation_method = "simple_zone2_reach"
        s_zone2_impedance_basis = "simple_selected_downstream_branch_impedance"

    # -------------------------------------------------------------------------
    # Zone 3: Longest Valid Downstream Branch (Now Parallel Corridor Proofed)
    # -------------------------------------------------------------------------
    d_zone3_branch = select_zone3_longest_valid_downstream_branch(
        d_corridor,
        l_all_corridors=l_all_corridors
    )

    f_zone3_branch_r = float(d_zone3_branch.get("Branch Resistance", 0.0) or 0.0)
    f_zone3_branch_x = float(d_zone3_branch.get("Branch Reactance", 0.0) or 0.0)
    f_zone3_branch_length = float(d_zone3_branch.get("Branch Length", 0.0) or 0.0)
    s_zone3_branch_id = join_line_names(d_zone3_branch.get("Branch Lines", []))

    f_zone3_r_reach = round(f_protected_corridor_r + (f_zone3_reach_factor * f_zone3_branch_r), 3)
    f_zone3_x_reach = round(f_protected_corridor_x + (f_zone3_reach_factor * f_zone3_branch_x), 3)

    return {
        "protected_corridor_r_ohm": round(f_protected_corridor_r, 3),
        "protected_corridor_x_ohm": round(f_protected_corridor_x, 3),
        "protected_corridor_length_km": round(f_protected_corridor_length, 3),
        "zone1_r_reach_ohm": f_zone1_r_reach,
        "zone1_x_reach_ohm": f_zone1_x_reach,
        "zone2_r_reach_ohm": f_zone2_r_reach,
        "zone2_x_reach_ohm": f_zone2_x_reach,
        "intended_zone2_r_reach_ohm": f_intended_zone2_r_reach,
        "intended_zone2_x_reach_ohm": f_intended_zone2_x_reach,
        "zone2_reach_calculation_method": s_zone2_reach_calculation_method,
        "zone2_branch_selection_method": d_zone2_selection.get("zone2_branch_selection_method", ""),
        "zone2_selected_remote_busbar": d_zone2_selection.get("selected_remote_busbar_name", ""),
        "zone2_downstream_branch_id": s_zone2_branch_id,
        "zone2_downstream_branch_length_km": round(f_zone2_selected_branch_length, 3),
        "zone2_downstream_branch_r_ohm": round(f_zone2_selected_branch_r, 3),
        "zone2_downstream_branch_x_ohm": round(f_zone2_selected_branch_x, 3),
        "zone2_selected_branch_has_parallel": get_boolean_value(b_zone2_has_parallel),
        "zone2_selected_branch_parallel_count": i_zone2_parallel_count,
        "zone2_impedance_basis": s_zone2_impedance_basis,
        "zone2_branch_for_reach_r_ohm": round(f_zone2_selected_branch_r, 3),
        "zone2_branch_for_reach_x_ohm": round(f_zone2_selected_branch_x, 3),
        "zone2_parallel_branch_for_complex_id": s_zone2_parallel_branch_id,
        "zone2_parallel_branch_for_complex_length_km": round(f_zone2_parallel_branch_length, 3),
        "zone2_parallel_branch_for_complex_r_ohm": round(f_zone2_parallel_branch_r, 3),
        "zone2_parallel_branch_for_complex_x_ohm": round(f_zone2_parallel_branch_x, 3),
        "zone3_r_reach_ohm": f_zone3_r_reach,
        "zone3_x_reach_ohm": f_zone3_x_reach,
        "zone3_branch_selection_method": "longest_valid_downstream_branch_excluding_relay_return",
        "zone3_downstream_branch_id": s_zone3_branch_id,
        "zone3_downstream_branch_length_km": round(f_zone3_branch_length, 3),
        "zone3_downstream_branch_r_ohm": round(f_zone3_branch_r, 3),
        "zone3_downstream_branch_x_ohm": round(f_zone3_branch_x, 3),
        "protected_corridor_is_parallel": get_boolean_value(bool(d_parallel_summary.get("is_parallel", False))),
        "protected_corridor_parallel_count": int(d_parallel_summary.get("parallel_count", 1) or 1),
        "protected_corridor_parallel_alternative_lines": d_parallel_summary.get("parallel_alternative_lines", ""),
    }


def split_protected_corridor_turbines_by_zone_reach(
        d_corridor,
        l_turbines,
        f_zone1_reach_fraction,
):
    """
    Splits protected-corridor turbines into:
        - Zone 1 turbines: located before or at Zone 1 reach point
        - Zone 2 turbines: located after Zone 1 reach point
    """

    l_zone1_turbines = []
    l_zone2_turbines = []

    o_relay_busbar = d_corridor.get(
        "relay_busbar",
    )

    l_protected_lines = d_corridor.get(
        "line_sections",
        [],
    )

    l_path_segments = build_ordered_path_segments(
        o_relay_busbar,
        l_protected_lines,
    )

    if not l_path_segments:
        return l_zone1_turbines, l_zone2_turbines

    f_total_path_length_km = l_path_segments[-1].get(
        "end_distance_km",
        0.0,
    )

    f_zone1_reach_distance_km = (
            float(f_zone1_reach_fraction or 0.0)
            * f_total_path_length_km
    )

    for o_dg in l_turbines:
        o_dg_terminal = get_connected_terminal_for_distributed_generator(
            o_dg,
        )

        f_dg_distance_km = get_terminal_distance_on_ordered_path(
            o_dg_terminal,
            l_path_segments,
        )

        if f_dg_distance_km is None:
            continue

        if f_dg_distance_km <= f_zone1_reach_distance_km + 1e-9:
            l_zone1_turbines.append(
                o_dg,
            )
        else:
            l_zone2_turbines.append(
                o_dg,
            )

    return (
        get_unique_objects(l_zone1_turbines),
        get_unique_objects(l_zone2_turbines),
    )


def get_branch_line_names(l_lines):
    return "; ".join([
        _get_safe_name(o_line)
        for o_line in l_lines
        if o_line is not None
    ])


def get_branch_id_from_branch(d_branch):
    return get_branch_line_names(
        d_branch.get(
            "Branch Lines",
            [],
        )
    )


def get_branch_remote_node_id(d_branch):
    l_terminals = d_branch.get(
        "Branch Terminals",
        [],
    )

    if not l_terminals:
        return ""

    return _get_safe_name(
        l_terminals[-1],
    )


def get_branch_hop_count(d_branch):
    return len(
        d_branch.get(
            "Branch Lines",
            [],
        )
    )


def get_branch_junction_node_count(d_branch):
    l_terminals = d_branch.get(
        "Branch Terminals",
        [],
    )

    return sum([
        1
        for o_terminal in l_terminals
        if get_terminal_is_junction_node(o_terminal)
    ])


def get_branch_distributed_generation_count(d_branch):
    l_terminals = d_branch.get(
        "Branch Terminals",
        [],
    )

    l_dg = get_unique_distributed_generators_from_terminals(
        l_terminals,
    )

    return len(l_dg)


def get_parallel_lines_for_line(o_line):
    """
    Returns physical parallel lines directly connecting the same two terminals.
    """

    if o_line is None:
        return []

    o_terminal_1, o_terminal_2, _, _ = get_line_terminals(
        o_line,
    )

    if o_terminal_1 is None or o_terminal_2 is None:
        return []

    s_terminal_1_full_name = _get_safe_full_name(
        o_terminal_1,
    )

    s_terminal_2_full_name = _get_safe_full_name(
        o_terminal_2,
    )

    l_parallel_lines = []

    for o_candidate_line in get_terminal_connected_lines(o_terminal_1):
        if o_candidate_line == o_line:
            continue

        if line_connects_terminal_pair_by_full_name(
                o_candidate_line,
                s_terminal_1_full_name,
                s_terminal_2_full_name,
        ):
            l_parallel_lines.append(
                o_candidate_line,
            )

    return get_unique_objects(
        l_parallel_lines,
    )


def get_parallel_lines_for_line_list(l_lines):
    l_parallel_lines = []

    for o_line in l_lines:
        l_parallel_lines.extend(
            get_parallel_lines_for_line(
                o_line,
            )
        )

    return get_unique_objects(
        l_parallel_lines,
    )


def get_parallel_line_names_for_line_list(l_lines):
    return get_branch_line_names(
        get_parallel_lines_for_line_list(
            l_lines,
        )
    )


def get_physical_corridor_line_key(d_corridor):
    return " || ".join(
        sorted([
            _get_safe_full_name(o_line)
            for o_line in d_corridor.get(
                "line_sections",
                [],
            )
            if o_line is not None
        ])
    )


def get_parallel_protected_corridor_line_names(
        d_corridor,
        l_all_corridors,
):
    """
    Returns unique physical parallel protected-corridor lines.

    Directional duplicates are ignored.
    """

    o_relay_busbar = d_corridor.get(
        "relay_busbar",
    )

    o_subsequent_busbar = d_corridor.get(
        "subsequent_busbar",
    )

    if o_relay_busbar is None or o_subsequent_busbar is None:
        return ""

    s_target_busbar_pair_key = frozenset([
        _get_safe_full_name(o_relay_busbar),
        _get_safe_full_name(o_subsequent_busbar),
    ])

    s_current_physical_key = get_physical_corridor_line_key(
        d_corridor,
    )

    d_parallel_names_by_key = {}

    for d_candidate_corridor in l_all_corridors:
        o_candidate_relay_busbar = d_candidate_corridor.get(
            "relay_busbar",
        )

        o_candidate_subsequent_busbar = d_candidate_corridor.get(
            "subsequent_busbar",
        )

        if o_candidate_relay_busbar is None or o_candidate_subsequent_busbar is None:
            continue

        s_candidate_busbar_pair_key = frozenset([
            _get_safe_full_name(o_candidate_relay_busbar),
            _get_safe_full_name(o_candidate_subsequent_busbar),
        ])

        if s_candidate_busbar_pair_key != s_target_busbar_pair_key:
            continue

        s_candidate_physical_key = get_physical_corridor_line_key(
            d_candidate_corridor,
        )

        if not s_candidate_physical_key:
            continue

        if s_candidate_physical_key == s_current_physical_key:
            continue

        d_parallel_names_by_key[s_candidate_physical_key] = join_line_names(
            d_candidate_corridor.get(
                "line_sections",
                [],
            )
        )

    return "; ".join(
        sorted(
            d_parallel_names_by_key.values(),
        )
    )


def summarize_next_node_downstream_context(l_downstream_branches):
    """
    Summarises all valid downstream branches from the subsequent node.

    This fixes the old behaviour where only the first connected element was
    considered.
    """

    l_remote_busbars = []
    l_junction_nodes = []
    l_dg = []

    for d_branch in l_downstream_branches:
        l_branch_terminals = d_branch.get(
            "Branch Terminals",
            [],
        )

        if l_branch_terminals:
            o_remote_terminal = l_branch_terminals[-1]

            if get_terminal_is_busbar(
                    o_remote_terminal,
            ):
                l_remote_busbars.append(
                    o_remote_terminal,
                )

        for o_terminal in l_branch_terminals:
            if get_terminal_is_junction_node(
                    o_terminal,
            ):
                l_junction_nodes.append(
                    o_terminal,
                )

        l_dg.extend(
            get_unique_distributed_generators_from_terminals(
                l_branch_terminals,
            )
        )

    return {
        "busbar_count": len(
            get_unique_objects(l_remote_busbars),
        ),
        "junction_node_count": len(
            get_unique_objects(l_junction_nodes),
        ),
        "distributed_generation_count": len(
            get_unique_objects(l_dg),
        ),
    }


# =============================================================================
# Feature Row Construction
# =============================================================================
def get_object_id_string(l_objects):
    """
    Returns readable object IDs for traceability columns.
    """

    return " -> ".join([
        _get_safe_name(o_object)
        for o_object in get_unique_objects(l_objects)
        if o_object is not None
    ])


def get_dg_summary_id(d_dg_summary):
    """
    Returns DG IDs from a DG summary dictionary.
    """

    return d_dg_summary.get(
        "names",
        "",
    )


def get_dg_summary_capacity_mva(d_dg_summary):
    """
    Returns total DG capacity from a DG summary dictionary.
    """

    return d_dg_summary.get(
        "capacity_mva",
        0.0,
    )


def get_branch_junction_node_id(d_branch):
    """
    Returns junction-node IDs inside a branch.
    """

    l_branch_terminals = d_branch.get(
        "Branch Terminals",
        [],
    )

    l_junction_nodes = [
        o_terminal
        for o_terminal in l_branch_terminals
        if get_terminal_is_junction_node(o_terminal)
    ]

    return get_object_id_string(
        l_junction_nodes,
    )


def get_branch_distributed_generation_id(d_branch):
    """
    Returns distributed-generator IDs inside a branch.
    """

    l_branch_terminals = d_branch.get(
        "Branch Terminals",
        [],
    )

    l_dg = get_unique_distributed_generators_from_terminals(
        l_branch_terminals,
    )

    return get_object_id_string(
        l_dg,
    )


def get_case_feature_row_for_corridor(
        o_project,
        o_grid,
        d_corridor,
        d_network,
        l_protected_corridors,
        i_case_counter,
):
    """
    Constructs one final unified feature row for one protected corridor.

    Pipeline inside this function:
        1. Read already-extracted network data
        2. Calculate topology features
        3. Select Zone 2 downstream branch
        4. Select Zone 3 downstream branch
        5. Calculate Zone 1 / Zone 2 / Zone 3 base reaches
        6. Detect DG/turbines by corridor location
        7. Classify turbines into Zone 1 and Zone 2
        8. Calculate compact infeed summaries
        9. Calculate final target reaches
        10. Return one row matching l_case_feature_columns
    """

    # =========================================================================
    # 0. Read already-extracted data
    # =========================================================================
    # Keep old helper-function naming compatible.
    l_all_corridors = l_protected_corridors or []

    # =========================================================================
    # 1. Corridor identity
    # =========================================================================
    s_case_id = f"{i_case_counter:05d}"

    o_relay_busbar = d_corridor.get(
        "relay_busbar",
    )

    o_subsequent_busbar = d_corridor.get(
        "subsequent_busbar",
    )

    o_first_line = d_corridor.get(
        "first_line_section",
    )

    o_relay_cubicle = d_corridor.get(
        "relay_cubicle",
    )

    l_protected_lines = d_corridor.get(
        "line_sections",
        [],
    )

    s_protected_corridor_id = d_corridor.get(
        "protected_corridor_id",
        "",
    )

    s_relay_node_id = _get_safe_name(
        o_relay_busbar,
    )

    s_subsequent_node_id = _get_safe_name(
        o_subsequent_busbar,
    )

    s_relay_id = get_relay_id_from_terminal_and_line(
        o_relay_busbar,
        o_first_line,
    )

    # =========================================================================
    # 2. Protected corridor topology and electrical data
    # =========================================================================
    f_protected_corridor_length_km = d_corridor.get(
        "total_length_km",
        0.0,
    )

    f_protected_corridor_r_ohm = d_corridor.get(
        "total_r_ohm",
        0.0,
    )

    f_protected_corridor_x_ohm = d_corridor.get(
        "total_x_ohm",
        0.0,
    )

    i_corridor_hop_count = d_corridor.get(
        "hop_count",
        0,
    )

    i_line_is_in_service = 1

    d_parallel_summary = summarize_parallel_protected_corridors(
        d_corridor,
        l_all_corridors,
    )

    s_protected_corridor_parallel_lines = get_parallel_protected_corridor_line_names(
        d_corridor=d_corridor,
        l_all_corridors=l_all_corridors,
    )

    i_protected_corridor_is_parallel = get_boolean_value(
        bool(
            s_protected_corridor_parallel_lines,
        )
    )

    # =========================================================================
    # 3. Subsequent-node / next-node summary
    # =========================================================================
    l_valid_downstream_branches = filter_valid_downstream_branches_excluding_relay_return(
        d_corridor=d_corridor,
        l_all_corridors=l_all_corridors,
    )

    d_next_node_downstream_context = summarize_next_node_downstream_context(
        l_valid_downstream_branches,
    )

    i_next_node_busbar_count = d_next_node_downstream_context.get(
        "busbar_count",
        0,
    )

    i_next_node_junction_node_count = d_next_node_downstream_context.get(
        "junction_node_count",
        0,
    )

    i_next_node_distributed_generation_count = d_next_node_downstream_context.get(
        "distributed_generation_count",
        0,
    )

    # =========================================================================
    # 4. Zone 2 downstream branch selection
    # =========================================================================
    d_zone2_selection = select_zone2_downstream_branch_group(
        d_corridor,
        l_all_corridors=l_all_corridors,
    )

    d_zone2_downstream_branch = d_zone2_selection.get(
        "selected_branch",
        build_empty_branch_summary(
            o_subsequent_busbar,
        ),
    )

    s_shortest_downstream_branch_id = d_zone2_downstream_branch.get(
        "Branch ID",
        join_line_names(
            d_zone2_downstream_branch.get(
                "Branch Lines",
                [],
            )
        ),
    )

    s_shortest_downstream_branch_remote_node_id = d_zone2_downstream_branch.get(
        "Remote Node ID",
        "",
    )

    i_shortest_downstream_branch_hop_count = int(
        d_zone2_downstream_branch.get(
            "Hop Count",
            len(
                d_zone2_downstream_branch.get(
                    "Branch Lines",
                    [],
                )
            ),
        ) or 0
    )

    s_shortest_downstream_branch_parallel_lines = d_zone2_downstream_branch.get(
        "Parallel Lines",
        "",
    )

    i_shortest_downstream_branch_junction_node_count = int(
        d_zone2_downstream_branch.get(
            "Junction Node Count",
            0,
        ) or 0
    )

    i_shortest_downstream_branch_distributed_generation_count = int(
        d_zone2_downstream_branch.get(
            "Distributed Generation Count",
            0,
        ) or 0
    )

    f_zone2_downstream_branch_length_km = d_zone2_downstream_branch.get(
        "Branch Length",
        0.0,
    )

    f_zone2_downstream_branch_r_ohm = d_zone2_downstream_branch.get(
        "Branch Resistance",
        0.0,
    )

    f_zone2_downstream_branch_x_ohm = d_zone2_downstream_branch.get(
        "Branch Reactance",
        0.0,
    )

    i_zone2_selected_branch_has_parallel = get_boolean_value(
        d_zone2_downstream_branch.get(
            "Has Parallel",
            False,
        )
    )

    # =========================================================================
    # 5. Forward parallel-branch diagnostics
    # =========================================================================
    i_parallel_group_count_forward, _ = count_forward_parallel_branch_groups_for_corridor(
        d_corridor,
        l_all_corridors=l_all_corridors,
    )

    # =========================================================================
    # 6. Zone 3 downstream branch selection
    # =========================================================================
    d_zone3_downstream_branch = select_zone3_longest_valid_downstream_branch(
        d_corridor,
        l_all_corridors=l_all_corridors,
    )

    if d_zone3_downstream_branch is None:
        d_zone3_downstream_branch = build_empty_branch_summary(
            o_subsequent_busbar,
        )

    s_longest_downstream_branch_id = d_zone3_downstream_branch.get(
        "Branch ID",
        join_line_names(
            d_zone3_downstream_branch.get(
                "Branch Lines",
                [],
            )
        ),
    )

    i_longest_downstream_branch_hop_count = int(
        d_zone3_downstream_branch.get(
            "Hop Count",
            len(
                d_zone3_downstream_branch.get(
                    "Branch Lines",
                    [],
                )
            ),
        ) or 0
    )

    s_zone3_downstream_branch_id = join_line_names(
        d_zone3_downstream_branch.get(
            "Branch Lines",
            [],
        )
    )

    f_zone3_downstream_branch_length_km = d_zone3_downstream_branch.get(
        "Branch Length",
        0.0,
    )

    f_zone3_downstream_branch_r_ohm = d_zone3_downstream_branch.get(
        "Branch Resistance",
        0.0,
    )

    f_zone3_downstream_branch_x_ohm = d_zone3_downstream_branch.get(
        "Branch Reactance",
        0.0,
    )

    if d_zone3_downstream_branch.get("Branch Lines", []):
        s_zone3_branch_selection_method = "longest_valid_downstream_branch"
    else:
        s_zone3_branch_selection_method = "no_valid_downstream_branch"

    # =========================================================================
    # 7. Distance zone reach calculation using existing stabilized engine
    # =========================================================================
    d_zone = calculate_distance_zone_reaches_for_corridor(
        d_corridor,
        d_zone2_downstream_branch,
        d_parallel_summary,
        l_all_corridors=l_all_corridors,
    )

    s_zone2_branch_selection_method = d_zone.get(
        "zone2_branch_selection_method",
        d_zone2_selection.get(
            "zone2_branch_selection_method",
            "",
        ),
    )

    s_zone2_reach_calculation_method = d_zone.get(
        "zone2_reach_calculation_method",
        d_zone2_selection.get(
            "zone2_reach_calculation_method",
            "",
        ),
    )

    s_zone2_downstream_branch_id = d_zone.get(
        "zone2_downstream_branch_id",
        "",
    )

    if not s_zone2_downstream_branch_id:
        s_zone2_downstream_branch_id = join_line_names(
            d_zone2_downstream_branch.get(
                "Branch Lines",
                [],
            )
        )

    s_zone2_impedance_basis = d_zone.get(
        "zone2_impedance_basis",
        "",
    )

    s_zone2_parallel_branch_for_complex_id = d_zone.get(
        "zone2_parallel_branch_for_complex_id",
        "",
    )

    f_zone2_parallel_branch_for_complex_length_km = d_zone.get(
        "zone2_parallel_branch_for_complex_length_km",
        0.0,
    )

    f_zone2_parallel_branch_for_complex_r_ohm = d_zone.get(
        "zone2_parallel_branch_for_complex_r_ohm",
        0.0,
    )

    f_zone2_parallel_branch_for_complex_x_ohm = d_zone.get(
        "zone2_parallel_branch_for_complex_x_ohm",
        0.0,
    )

    s_zone3_branch_selection_method = d_zone.get(
        "zone3_branch_selection_method",
        s_zone3_branch_selection_method,
    )

    s_zone3_downstream_branch_id = d_zone.get(
        "zone3_downstream_branch_id",
        s_zone3_downstream_branch_id,
    )

    f_zone3_downstream_branch_length_km = d_zone.get(
        "zone3_downstream_branch_length_km",
        f_zone3_downstream_branch_length_km,
    )

    f_zone3_downstream_branch_r_ohm = d_zone.get(
        "zone3_downstream_branch_r_ohm",
        f_zone3_downstream_branch_r_ohm,
    )

    f_zone3_downstream_branch_x_ohm = d_zone.get(
        "zone3_downstream_branch_x_ohm",
        f_zone3_downstream_branch_x_ohm,
    )

    s_longest_downstream_branch_id = s_zone3_downstream_branch_id

    if s_longest_downstream_branch_id:
        i_longest_downstream_branch_hop_count = len([
            s_line_name
            for s_line_name in s_longest_downstream_branch_id.split(";")
            if s_line_name.strip()
        ])
    else:
        i_longest_downstream_branch_hop_count = 0

    # =========================================================================
    # 8. Base reach labels
    # =========================================================================
    f_zone1_base_r_reach_ohm = d_zone.get(
        "zone1_r_reach_ohm",
        round(
            i_reach_gf * f_protected_corridor_r_ohm,
            3,
        ),
    )

    f_zone1_base_x_reach_ohm = d_zone.get(
        "zone1_x_reach_ohm",
        round(
            i_reach_gf * f_protected_corridor_x_ohm,
            3,
        ),
    )

    f_zone2_base_r_reach_ohm = d_zone.get(
        "zone2_r_reach_ohm",
        round(
            i_reach_gf * (
                f_protected_corridor_r_ohm
                + i_reach_gf * f_zone2_downstream_branch_r_ohm
            ),
            3,
        ),
    )

    f_zone2_base_x_reach_ohm = d_zone.get(
        "zone2_x_reach_ohm",
        round(
            i_reach_gf * (
                f_protected_corridor_x_ohm
                + i_reach_gf * f_zone2_downstream_branch_x_ohm
            ),
            3,
        ),
    )

    f_zone3_base_r_reach_ohm = d_zone.get(
        "zone3_r_reach_ohm",
        round(
            f_zone3_reach_factor * (
                f_protected_corridor_r_ohm
                + f_zone3_downstream_branch_r_ohm
            ),
            3,
        ),
    )

    f_zone3_base_x_reach_ohm = d_zone.get(
        "zone3_x_reach_ohm",
        round(
            f_zone3_reach_factor * (
                f_protected_corridor_x_ohm
                + f_zone3_downstream_branch_x_ohm
            ),
            3,
        ),
    )

    # =========================================================================
    # 9. DG / turbine location detection
    # =========================================================================
    d_dg_context = summarize_dg_by_corridor_location(
        d_corridor,
        d_zone2_downstream_branch,
    )

    l_protected_corridor_turbines = d_dg_context["protected_corridor"].get(
        "objects",
        [],
    )

    l_zone1_turbines, l_protected_corridor_zone2_turbines = split_protected_corridor_turbines_by_zone_reach(
        d_corridor=d_corridor,
        l_turbines=l_protected_corridor_turbines,
        f_zone1_reach_fraction=i_reach_gf,
    )

    l_zone1_turbines = get_unique_objects(
        l_zone1_turbines,
    )

    l_zone2_turbines = []

    # Zone 2 must include Zone 1 turbines because they are also
    # intermediate infeeds for a Zone 2 fault.
    l_zone2_turbines.extend(
        l_zone1_turbines,
    )

    # Turbines on the protected corridor beyond Zone 1.
    l_zone2_turbines.extend(
        l_protected_corridor_zone2_turbines,
    )

    # Turbines at the subsequent busbar.
    l_zone2_turbines.extend(
        d_dg_context["subsequent_busbar"].get(
            "objects",
            [],
        )
    )

    # Turbines on the selected downstream branch.
    l_zone2_turbines.extend(
        d_dg_context["downstream_branch"].get(
            "objects",
            [],
        )
    )

    l_zone2_turbines = get_unique_objects(
        l_zone2_turbines,
    )

    # =========================================================================
    # 10. Relay reference Ikss
    # =========================================================================
    o_reference_cubicle, _, s_relay_ikss_line_attr, _ = _get_relay_side_and_cubicle(
        o_first_line,
        o_relay_busbar,
    )

    if o_reference_cubicle is None:
        o_reference_cubicle = o_relay_cubicle

    f_relay_reference_ikss_ka, _ = _read_relay_reference_ikss(
        o_reference_cubicle,
        o_first_line,
        s_relay_ikss_line_attr,
    )

    # =========================================================================
    # 11. Zone 1 / Zone 2 compact infeed summaries
    # =========================================================================
    # =========================================================================

    l_zone1_fault_lines = l_protected_lines[:]

    d_zone1_infeed_summary = calculate_zone_infeed_summary_for_turbines(
        o_project=o_project,
        o_grid=o_grid,
        o_start_terminal=o_relay_busbar,
        o_reference_cubicle=o_reference_cubicle,
        o_reference_line=o_first_line,
        s_relay_ikss_line_attr=s_relay_ikss_line_attr,
        l_turbines=l_zone1_turbines,
        l_fault_lines=l_zone1_fault_lines,
        f_fault_reach_r_ohm=f_zone1_base_r_reach_ohm,
        f_fault_reach_x_ohm=f_zone1_base_x_reach_ohm,
        f_relay_reference_ikss_ka=f_relay_reference_ikss_ka,
        s_zone_name="zone1",
    )

    # =========================================================================
    # Zone 2 SHC/infeed summary
    # =========================================================================

    l_zone2_fault_lines = []

    l_zone2_fault_lines.extend(
        l_protected_lines,
    )

    l_zone2_fault_lines.extend(
        d_zone2_downstream_branch.get(
            "Branch Lines",
            [],
        )
    )

    d_zone2_infeed_summary = calculate_zone_infeed_summary_for_turbines(
        o_project=o_project,
        o_grid=o_grid,
        o_start_terminal=o_relay_busbar,
        o_reference_cubicle=o_reference_cubicle,
        o_reference_line=o_first_line,
        s_relay_ikss_line_attr=s_relay_ikss_line_attr,
        l_turbines=l_zone2_turbines,
        l_fault_lines=l_zone2_fault_lines,
        f_fault_reach_r_ohm=f_zone2_base_r_reach_ohm,
        f_fault_reach_x_ohm=f_zone2_base_x_reach_ohm,
        f_relay_reference_ikss_ka=f_relay_reference_ikss_ka,
        s_zone_name="zone2",
    )

    f_zone1_infeed_correction_r_ohm = d_zone1_infeed_summary.get(
        "infeed_correction_r_ohm",
        0.0,
    )

    f_zone1_infeed_correction_x_ohm = d_zone1_infeed_summary.get(
        "infeed_correction_x_ohm",
        0.0,
    )

    f_zone2_infeed_correction_r_ohm = d_zone2_infeed_summary.get(
        "infeed_correction_r_ohm",
        0.0,
    )

    f_zone2_infeed_correction_x_ohm = d_zone2_infeed_summary.get(
        "infeed_correction_x_ohm",
        0.0,
    )

    # =========================================================================
    # 12. Final target reaches
    # =========================================================================
    f_target_zone1_r_reach_ohm = round(
        f_zone1_base_r_reach_ohm
        + f_zone1_infeed_correction_r_ohm,
        3,
    )

    f_target_zone1_x_reach_ohm = round(
        f_zone1_base_x_reach_ohm
        + f_zone1_infeed_correction_x_ohm,
        3,
    )

    f_target_zone2_r_reach_ohm = round(
        f_zone2_base_r_reach_ohm
        + f_zone2_infeed_correction_r_ohm,
        3,
    )

    f_target_zone2_x_reach_ohm = round(
        f_zone2_base_x_reach_ohm
        + f_zone2_infeed_correction_x_ohm,
        3,
    )

    # No separate Zone 3 infeed correction is included in the selected schema.
    f_target_zone3_r_reach_ohm = f_zone3_base_r_reach_ohm
    f_target_zone3_x_reach_ohm = f_zone3_base_x_reach_ohm

    # =========================================================================
    # 13. Final row matching new l_case_feature_columns
    # =========================================================================
    d_row = {
        # ---------------------------------------------------------------------
        # PART 1: Core Graph Topology & Primary Edge Features
        # ---------------------------------------------------------------------
        "case_id": s_case_id,
        "relay_id": s_relay_id,
        "relay_node_id": s_relay_node_id,
        "subsequent_node_id": s_subsequent_node_id,
        "protected_corridor_id": s_protected_corridor_id,
        "protected_corridor_length_km": f_protected_corridor_length_km,
        "protected_corridor_r_ohm": f_protected_corridor_r_ohm,
        "protected_corridor_x_ohm": f_protected_corridor_x_ohm,
        "corridor_hop_count": i_corridor_hop_count,
        "line_is_in_service": i_line_is_in_service,
        "protected_corridor_is_parallel": i_protected_corridor_is_parallel,
        "protected_corridor_parallel_count": d_parallel_summary.get(
            "parallel_count",
            1,
        ),
        "protected_corridor_parallel_id": s_protected_corridor_parallel_lines,

        "next_node_busbar_count": i_next_node_busbar_count,
        "next_node_junction_node_count": i_next_node_junction_node_count,
        "next_node_distributed_generation_count": i_next_node_distributed_generation_count,

        # ---------------------------------------------------------------------
        # PART 2: Downstream look-ahead path context
        # ---------------------------------------------------------------------
        "shortest_downstream_branch_id": s_shortest_downstream_branch_id,
        "shortest_downstream_branch_remote_node_id": s_shortest_downstream_branch_remote_node_id,
        "shortest_downstream_branch_hop_count": i_shortest_downstream_branch_hop_count,
        "shortest_downstream_branch_length_km": f_zone2_downstream_branch_length_km,
        "shortest_downstream_branch_r_ohm": f_zone2_downstream_branch_r_ohm,
        "shortest_downstream_branch_x_ohm": f_zone2_downstream_branch_x_ohm,

        "shortest_downstream_branch_has_parallel": i_zone2_selected_branch_has_parallel,
        "shortest_downstream_branch_parallel_count": int(
            d_zone2_downstream_branch.get(
                "Parallel Count",
                0,
            ) or 0
        ),
        "shortest_downstream_branch_parallel_id": s_shortest_downstream_branch_parallel_lines,

        "shortest_downstream_branch_junction_node_count": i_shortest_downstream_branch_junction_node_count,
        "shortest_downstream_branch_junction_node_id": get_branch_junction_node_id(
            d_zone2_downstream_branch,
        ),

        "shortest_downstream_branch_distributed_generation_count": i_shortest_downstream_branch_distributed_generation_count,
        "shortest_downstream_branch_distributed_generation_id": get_branch_distributed_generation_id(
            d_zone2_downstream_branch,
        ),

        "zone2_branch_selection_method": s_zone2_branch_selection_method,
        "zone2_selected_branch_id": s_zone2_downstream_branch_id,
        "zone2_reach_calculation_method": s_zone2_reach_calculation_method,
        "zone2_impedance_basis": s_zone2_impedance_basis,

        "parallel_group_count_forward": i_parallel_group_count_forward,
        "parallel_group_id": s_zone2_parallel_branch_for_complex_id,
        "zone2_parallel_branch_for_complex_length_km": f_zone2_parallel_branch_for_complex_length_km,
        "zone2_parallel_branch_for_complex_r_ohm": f_zone2_parallel_branch_for_complex_r_ohm,
        "zone2_parallel_branch_for_complex_x_ohm": f_zone2_parallel_branch_for_complex_x_ohm,

        "zone3_branch_selection_method": s_zone3_branch_selection_method,
        "zone3_selected_branch_id": s_longest_downstream_branch_id,
        "longest_downstream_branch_hop_count": i_longest_downstream_branch_hop_count,
        "longest_downstream_branch_length_km": f_zone3_downstream_branch_length_km,
        "longest_downstream_branch_r_ohm": f_zone3_downstream_branch_r_ohm,
        "longest_downstream_branch_x_ohm": f_zone3_downstream_branch_x_ohm,

        # ---------------------------------------------------------------------
        # PART 3: Spatial Distributed Generation Features
        # ---------------------------------------------------------------------
        "relay_busbar_distributed_generation_count": d_dg_context["relay_busbar"].get(
            "count",
            0,
        ),
        "relay_busbar_distributed_generation_id": get_dg_summary_id(
            d_dg_context["relay_busbar"],
        ),
        "relay_busbar_distributed_generation_capacity_mva": d_dg_context["relay_busbar"].get(
            "capacity_mva",
            0.0,
        ),

        "protected_corridor_distributed_generation_count": d_dg_context["protected_corridor"].get(
            "count",
            0,
        ),
        "protected_corridor_distributed_generation_id": get_dg_summary_id(
            d_dg_context["protected_corridor"],
        ),
        "protected_corridor_distributed_generation_capacity_mva": d_dg_context["protected_corridor"].get(
            "capacity_mva",
            0.0,
        ),

        "subsequent_busbar_distributed_generation_count": d_dg_context["subsequent_busbar"].get(
            "count",
            0,
        ),
        "subsequent_busbar_distributed_generation_id": get_dg_summary_id(
            d_dg_context["subsequent_busbar"],
        ),
        "subsequent_busbar_distributed_generation_capacity_mva": d_dg_context["subsequent_busbar"].get(
            "capacity_mva",
            0.0,
        ),

        "downstream_branch_distributed_generation_count": d_dg_context["downstream_branch"].get(
            "count",
            0,
        ),
        "downstream_branch_distributed_generation_id": get_dg_summary_id(
            d_dg_context["downstream_branch"],
        ),
        "downstream_branch_distributed_generation_capacity_mva": d_dg_context["downstream_branch"].get(
            "capacity_mva",
            0.0,
        ),

        "remote_busbar_distributed_generation_count": d_dg_context["remote_busbar"].get(
            "count",
            0,
        ),
        "remote_busbar_distributed_generation_id": get_dg_summary_id(
            d_dg_context["remote_busbar"],
        ),
        "remote_busbar_distributed_generation_capacity_mva": d_dg_context["remote_busbar"].get(
            "capacity_mva",
            0.0,
        ),
        # ---------------------------------------------------------------------
        # Zone 1 / Zone 2 turbine pool features
        # ---------------------------------------------------------------------
        "zone1_turbines_candidate_count": d_zone1_infeed_summary.get(
            "turbines_candidate_count",
            0,
        ),
        "zone1_turbines_candidate_id": d_zone1_infeed_summary.get(
            "turbines_candidate_id",
            "",
        ),
        "zone1_turbines_considered_count": d_zone1_infeed_summary.get(
            "turbines_considered_count",
            0,
        ),
        "zone1_turbines_considered_id": d_zone1_infeed_summary.get(
            "turbines_considered_id",
            "",
        ),
        "zone1_turbines_skipped_count": d_zone1_infeed_summary.get(
            "turbines_skipped_count",
            0,
        ),
        "zone1_turbines_skipped_id": d_zone1_infeed_summary.get(
            "turbines_skipped_id",
            "",
        ),
        "zone1_turbines_total_capacity_mva": d_zone1_infeed_summary.get(
            "turbines_total_capacity_mva",
            0.0,
        ),

        "zone2_turbines_candidate_count": d_zone2_infeed_summary.get(
            "turbines_candidate_count",
            0,
        ),
        "zone2_turbines_candidate_id": d_zone2_infeed_summary.get(
            "turbines_candidate_id",
            "",
        ),
        "zone2_turbines_considered_count": d_zone2_infeed_summary.get(
            "turbines_considered_count",
            0,
        ),
        "zone2_turbines_considered_id": d_zone2_infeed_summary.get(
            "turbines_considered_id",
            "",
        ),
        "zone2_turbines_skipped_count": d_zone2_infeed_summary.get(
            "turbines_skipped_count",
            0,
        ),
        "zone2_turbines_skipped_id": d_zone2_infeed_summary.get(
            "turbines_skipped_id",
            "",
        ),
        "zone2_turbines_total_capacity_mva": d_zone2_infeed_summary.get(
            "turbines_total_capacity_mva",
            0.0,
        ),

        # ---------------------------------------------------------------------
        # PART 4: Multitask Ground-Truth Prediction Labels
        # ---------------------------------------------------------------------
        "base_zone1_r_reach_ohm": f_zone1_base_r_reach_ohm,
        "base_zone1_x_reach_ohm": f_zone1_base_x_reach_ohm,
        "zone1_infeed_correction_r_ohm": f_zone1_infeed_correction_r_ohm,
        "zone1_infeed_correction_x_ohm": f_zone1_infeed_correction_x_ohm,
        "target_zone1_r_reach_ohm": f_target_zone1_r_reach_ohm,
        "target_zone1_x_reach_ohm": f_target_zone1_x_reach_ohm,

        "base_zone2_r_reach_ohm": f_zone2_base_r_reach_ohm,
        "base_zone2_x_reach_ohm": f_zone2_base_x_reach_ohm,
        "zone2_infeed_correction_r_ohm": f_zone2_infeed_correction_r_ohm,
        "zone2_infeed_correction_x_ohm": f_zone2_infeed_correction_x_ohm,
        "target_zone2_r_reach_ohm": f_target_zone2_r_reach_ohm,
        "target_zone2_x_reach_ohm": f_target_zone2_x_reach_ohm,

        "base_zone3_r_reach_ohm": f_zone3_base_r_reach_ohm,
        "base_zone3_x_reach_ohm": f_zone3_base_x_reach_ohm,
        "target_zone3_r_reach_ohm": f_target_zone3_r_reach_ohm,
        "target_zone3_x_reach_ohm": f_target_zone3_x_reach_ohm,

        "dataset_version": globals().get(
            "s_dataset_version",
            "v0.1",
        ),
        "dataset_export_type": globals().get(
            "s_dataset_export_type",
            "unified_corridor_feature_matrix",
        ),
    }

    # =========================================================================
    # 14. Force exact output schema/order
    # =========================================================================
    return {
        s_column: d_row.get(
            s_column,
            None,
        )
        for s_column in l_case_feature_columns
    }


# =============================================================================
# Controlled Known-Case Infeed Diagnostic
# =============================================================================

def find_corridor_by_terminal_names_and_edge(
        l_protected_corridors,
        s_relay_busbar_name,
        s_subsequent_busbar_name,
        s_protected_corridor_id,
):
    """
    Finds one corridor by relay busbar, remote busbar, and protected edge id.
    """

    for d_corridor in l_protected_corridors:
        if _get_safe_name(d_corridor.get("relay_busbar")) != s_relay_busbar_name:
            continue

        if _get_safe_name(d_corridor.get("subsequent_busbar")) != s_subsequent_busbar_name:
            continue

        if d_corridor.get("protected_corridor_id") != s_protected_corridor_id:
            continue

        return d_corridor

    return None


# =============================================================================
# Export Functions
# =============================================================================

def get_feature_matrix_output_directory(
        o_project,
        o_grid,
):
    """
    Creates and returns the output directory for the unified feature matrix.
    """

    s_project_name = _get_safe_name(
        o_project,
    )

    s_grid_name_clean = _get_safe_name(
        o_grid,
    )

    # Remove characters that are awkward in folder names.
    s_project_name = (
        s_project_name
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )

    s_grid_name_clean = (
        s_grid_name_clean
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )

    s_output_dir = os.path.join(
        s_dir_codingbase,
        f"{s_project_name}_{s_grid_name_clean}_Unified_Feature_Matrix",
    )

    return s_output_dir


def safe_percentage(
        f_numerator,
        f_denominator,
):
    """
    Safely calculates percentage.
    """

    try:
        f_numerator = float(
            f_numerator or 0.0,
        )

        f_denominator = float(
            f_denominator or 0.0,
        )

    except Exception:
        return 0.0

    if abs(f_denominator) <= f_zero_tolerance:
        return 0.0

    return round(
        100.0 * f_numerator / f_denominator,
        3,
    )


#Check only
def export_reach_percentage_diagnostics(
        o_project,
        o_grid,
        df_case_feature_matrix,
):
    """
    Exports reach percentage diagnostics.

    This does not modify the final ML dataset.
    It creates a separate diagnostics file.
    """

    l_rows = []

    for _, d_row in df_case_feature_matrix.iterrows():
        f_protected_x = d_row.get(
            "protected_corridor_x_ohm",
            0.0,
        )

        f_zone2_downstream_x = d_row.get(
            "shortest_downstream_branch_x_ohm",
            0.0,
        )

        f_zone3_downstream_x = d_row.get(
            "longest_downstream_branch_x_ohm",
            d_row.get(
                "zone3_downstream_branch_x_ohm",
                0.0,
            ),
        )

        f_zone2_path_x = (
                f_protected_x
                + f_zone2_downstream_x
        )

        f_zone3_path_x = (
                f_protected_x
                + f_zone3_downstream_x
        )

        d_percentage_row = {
            "case_id": d_row.get(
                "case_id",
                "",
            ),
            "relay_node_id": d_row.get(
                "relay_node_id",
                "",
            ),
            "subsequent_node_id": d_row.get(
                "subsequent_node_id",
                "",
            ),
            "protected_corridor_id": d_row.get(
                "protected_corridor_id",
                "",
            ),

            "protected_corridor_x_ohm": round(
                float(f_protected_x or 0.0),
                6,
            ),
            "zone2_path_x_ohm": round(
                float(f_zone2_path_x or 0.0),
                6,
            ),
            "zone3_path_x_ohm": round(
                float(f_zone3_path_x or 0.0),
                6,
            ),

            # Percentage relative to protected corridor only.
            "zone1_base_percent_of_protected_x": safe_percentage(
                d_row.get("base_zone1_x_reach_ohm", 0.0),
                f_protected_x,
            ),
            "zone1_target_percent_of_protected_x": safe_percentage(
                d_row.get("target_zone1_x_reach_ohm", 0.0),
                f_protected_x,
            ),

            "zone2_base_percent_of_protected_x": safe_percentage(
                d_row.get("base_zone2_x_reach_ohm", 0.0),
                f_protected_x,
            ),
            "zone2_target_percent_of_protected_x": safe_percentage(
                d_row.get("target_zone2_x_reach_ohm", 0.0),
                f_protected_x,
            ),

            "zone3_base_percent_of_protected_x": safe_percentage(
                d_row.get("base_zone3_x_reach_ohm", 0.0),
                f_protected_x,
            ),
            "zone3_target_percent_of_protected_x": safe_percentage(
                d_row.get("target_zone3_x_reach_ohm", 0.0),
                f_protected_x,
            ),

            # Percentage relative to selected total path.
            "zone2_base_percent_of_zone2_path_x": safe_percentage(
                d_row.get("base_zone2_x_reach_ohm", 0.0),
                f_zone2_path_x,
            ),
            "zone2_target_percent_of_zone2_path_x": safe_percentage(
                d_row.get("target_zone2_x_reach_ohm", 0.0),
                f_zone2_path_x,
            ),

            "zone3_base_percent_of_zone3_path_x": safe_percentage(
                d_row.get("base_zone3_x_reach_ohm", 0.0),
                f_zone3_path_x,
            ),
            "zone3_target_percent_of_zone3_path_x": safe_percentage(
                d_row.get("target_zone3_x_reach_ohm", 0.0),
                f_zone3_path_x,
            ),
        }

        l_rows.append(
            d_percentage_row,
        )

    df_reach_percentage_diagnostics = pd.DataFrame(
        l_rows,
    )

    s_output_dir = get_feature_matrix_output_directory(
        o_project,
        o_grid,
    )

    os.makedirs(
        s_output_dir,
        exist_ok=True,
    )

    s_csv_path = os.path.join(
        s_output_dir,
        "reach_percentage_diagnostics.csv",
    )

    s_excel_path = os.path.join(
        s_output_dir,
        "reach_percentage_diagnostics.xlsx",
    )

    df_reach_percentage_diagnostics.to_csv(
        s_csv_path,
        index=False,
    )

    df_reach_percentage_diagnostics.to_excel(
        s_excel_path,
        index=False,
    )

    _print(
        "\n✅ Reach percentage diagnostics exported."
    )

    _print(
        f"CSV:   {s_csv_path}"
    )

    _print(
        f"Excel: {s_excel_path}"
    )

    return df_reach_percentage_diagnostics


def validate_no_missing_export_columns(
        df_case_feature_matrix,
):
    """
    Checks whether any selected output column is completely empty.

    This is useful after changing l_case_feature_columns because pandas will
    still export a column even if every row has None for that key.
    """

    l_problem_columns = []

    for s_column in l_case_feature_columns:
        if s_column not in df_case_feature_matrix.columns:
            l_problem_columns.append(
                s_column,
            )
            continue

        s_series = df_case_feature_matrix[s_column]

        b_all_missing = s_series.isna().all()

        b_all_empty_string = (
            s_series
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .all()
        )

        if b_all_missing or b_all_empty_string:
            l_problem_columns.append(
                s_column,
            )

    if not l_problem_columns:
        _print(
            "\n✅ Output schema validation passed: no fully empty selected columns."
        )
        return

    _print(
        "⚠️ Output schema validation warning. These selected columns are fully empty:"
    )

    for s_column in l_problem_columns:
        _print(
            f"   - {s_column}"
        )


def export_unified_corridor_feature_matrix(
        o_project,
        o_grid,
):
    _print("\n⚙️ Creating unified corridor-based feature matrix...")

    d_network = extract_network_once(
        o_grid,
    )

    l_protected_corridors = detect_protected_corridors_once(
        d_network,
    )

    l_protected_corridors = sorted(
        l_protected_corridors,
        key=lambda d_corridor: (
            _get_safe_name(
                d_corridor.get(
                    "relay_busbar",
                )
            ),
            _get_safe_name(
                d_corridor.get(
                    "subsequent_busbar",
                )
            ),
            join_line_names(
                d_corridor.get(
                    "line_sections",
                    [],
                )
            ),
        ),
    )

    l_feature_rows = []

    for i_case_counter, d_corridor in enumerate(l_protected_corridors, start=1):
        s_relay_node_id = _get_safe_name(
            d_corridor.get(
                "relay_busbar",
            )
        )

        s_subsequent_node_id = _get_safe_name(
            d_corridor.get(
                "subsequent_busbar",
            )
        )

        s_protected_corridor_id = d_corridor.get(
            "protected_corridor_id",
            "",
        )

        _print(
            f"\n✅ {i_case_counter:03d}. "
            f"{s_relay_node_id} → "
            f"{s_subsequent_node_id} | "
            f"{s_protected_corridor_id}"
        )

        d_row = get_case_feature_row_for_corridor(
            o_project=o_project,
            o_grid=o_grid,
            d_corridor=d_corridor,
            d_network=d_network,
            l_protected_corridors=l_protected_corridors,
            i_case_counter=i_case_counter,
        )

        l_feature_rows.append(
            d_row,
        )

        _print(
            f"   Result: "
            f"Z1=({d_row.get('target_zone1_r_reach_ohm', 0.0)}, "
            f"{d_row.get('target_zone1_x_reach_ohm', 0.0)}) | "
            f"Z2=({d_row.get('target_zone2_r_reach_ohm', 0.0)}, "
            f"{d_row.get('target_zone2_x_reach_ohm', 0.0)}) | "
            f"Z3=({d_row.get('target_zone3_r_reach_ohm', 0.0)}, "
            f"{d_row.get('target_zone3_x_reach_ohm', 0.0)})"
        )

    df_case_feature_matrix = pd.DataFrame(
        l_feature_rows,
        columns=l_case_feature_columns,
    )

    # Check only
    validate_no_missing_export_columns(
        df_case_feature_matrix,
    )

    #Check only
    validate_reach_correction_addition(
        df_case_feature_matrix,
    )

    # Check only
    export_reach_percentage_diagnostics(
        o_project=o_project,
        o_grid=o_grid,
        df_case_feature_matrix=df_case_feature_matrix,
    )

    s_output_dir = get_feature_matrix_output_directory(
        o_project,
        o_grid,
    )

    os.makedirs(
        s_output_dir,
        exist_ok=True,
    )

    s_csv_path = os.path.join(
        s_output_dir,
        "case_feature_matrix_unified.csv",
    )

    s_excel_path = os.path.join(
        s_output_dir,
        "case_feature_matrix_unified.xlsx",
    )

    df_case_feature_matrix.to_csv(
        s_csv_path,
        index=False,
    )

    df_case_feature_matrix.to_excel(
        s_excel_path,
        index=False,
    )

    _print("\n✅ Unified corridor feature matrix exported.")
    _print(f"Rows:  {len(df_case_feature_matrix)}")
    _print(f"CSV:   {s_csv_path}")
    _print(f"Excel: {s_excel_path}")

    return df_case_feature_matrix


def export_ml_ready_feature_matrix_from_dataframe(
        df_feature_matrix,
        s_output_dir,
        s_dataset_base_filename,
):
    """
    Exports an ML-ready version of the feature matrix.

    The full engineering dataset keeps trace/debug columns.
    This ML-ready export removes:
        - object IDs and names
        - scenario/case identifiers
        - branch-selection method strings
        - random seed / scale metadata
        - intermediate infeed correction labels

    It keeps:
        - numeric topology/electrical/DG features
        - final target reach labels
    """

    l_final_target_columns = [
        "target_zone1_r_reach_ohm",
        "target_zone1_x_reach_ohm",
        "target_zone2_r_reach_ohm",
        "target_zone2_x_reach_ohm",
        "target_zone3_r_reach_ohm",
        "target_zone3_x_reach_ohm",
    ]

    l_infeed_correction_label_columns = [
        "zone1_infeed_correction_r_ohm",
        "zone1_infeed_correction_x_ohm",
        "zone2_infeed_correction_r_ohm",
        "zone2_infeed_correction_x_ohm",
    ]

    l_trace_index_columns = [
        "scenario_id",
        "case_uid",
        "case_id",

        "switch_state_short_id",
        "switch_state_config_id",
        "switch_state_row_index",

        "relay_id",
        "relay_node_id",
        "protected_corridor_id",
        "subsequent_node_id",

        "zone2_selected_branch_id",
        "zone3_selected_branch_id",

        "line_length_random_seed",
        "line_length_scale_min",
        "line_length_scale_max",

        "dg_capacity_random_seed",
        "dg_capacity_scale_min",
        "dg_capacity_scale_max",
    ]

    l_missing_target_columns = [
        s_column
        for s_column in l_final_target_columns
        if s_column not in df_feature_matrix.columns
    ]

    if l_missing_target_columns:
        raise RuntimeError(
            "❌ ML-ready export failed. Missing target columns: "
            + ", ".join(l_missing_target_columns)
        )

    # -------------------------------------------------------------------------
    # Export trace index separately.
    # This lets you map each ML row back to the engineering dataset.
    # -------------------------------------------------------------------------
    l_existing_trace_index_columns = [
        s_column
        for s_column in l_trace_index_columns
        if s_column in df_feature_matrix.columns
    ]

    df_trace_index = df_feature_matrix[
        l_existing_trace_index_columns
    ].copy()

    s_trace_csv_path = os.path.join(
        s_output_dir,
        f"{s_dataset_base_filename}_trace_index.csv",
    )

    s_trace_excel_path = os.path.join(
        s_output_dir,
        f"{s_dataset_base_filename}_trace_index.xlsx",
    )

    df_trace_index.to_csv(
        s_trace_csv_path,
        index=False,
    )

    df_trace_index.to_excel(
        s_trace_excel_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Decide which columns must be removed from ML input.
    # -------------------------------------------------------------------------
    l_columns_to_drop = []

    for s_column in df_feature_matrix.columns:
        b_is_target_column = s_column in l_final_target_columns

        if b_is_target_column:
            continue

        if s_column in l_infeed_correction_label_columns:
            l_columns_to_drop.append(
                s_column,
            )
            continue

        if s_column in [
            "case_uid",
            "relay_id",

            "line_length_random_seed",
            "line_length_scale_min",
            "line_length_scale_max",

            "dg_capacity_random_seed",
            "dg_capacity_scale_min",
            "dg_capacity_scale_max",
        ]:
            l_columns_to_drop.append(
                s_column,
            )
            continue

        if s_column.endswith(
                "_id",
        ):
            l_columns_to_drop.append(
                s_column,
            )
            continue

        if s_column.endswith(
                "_method",
        ):
            l_columns_to_drop.append(
                s_column,
            )
            continue

        if s_column.endswith(
                "_basis",
        ):
            l_columns_to_drop.append(
                s_column,
            )
            continue

    df_ml_ready = df_feature_matrix.drop(
        columns=l_columns_to_drop,
        errors="ignore",
    ).copy()

    # -------------------------------------------------------------------------
    # Convert all remaining columns to numeric.
    # Any non-numeric leftover column is dropped.
    # -------------------------------------------------------------------------
    l_non_numeric_columns_to_drop = []

    for s_column in df_ml_ready.columns:
        df_ml_ready[s_column] = pd.to_numeric(
            df_ml_ready[s_column],
            errors="coerce",
        )

        if (
                s_column not in l_final_target_columns
                and df_ml_ready[s_column].isna().all()
        ):
            l_non_numeric_columns_to_drop.append(
                s_column,
            )

    if l_non_numeric_columns_to_drop:
        df_ml_ready = df_ml_ready.drop(
            columns=l_non_numeric_columns_to_drop,
            errors="ignore",
        )

    # -------------------------------------------------------------------------
    # Fill missing feature values with 0.0.
    # Targets must not be missing.
    # -------------------------------------------------------------------------
    l_existing_target_columns = [
        s_column
        for s_column in l_final_target_columns
        if s_column in df_ml_ready.columns
    ]

    l_feature_columns = [
        s_column
        for s_column in df_ml_ready.columns
        if s_column not in l_existing_target_columns
    ]

    df_ml_ready[l_feature_columns] = df_ml_ready[l_feature_columns].fillna(
        0.0,
    )

    if df_ml_ready[l_existing_target_columns].isna().any().any():
        raise RuntimeError(
            "❌ ML-ready export failed. At least one target value is missing."
        )

    # -------------------------------------------------------------------------
    # Keep target columns at the end.
    # -------------------------------------------------------------------------
    df_ml_ready = df_ml_ready[
        l_feature_columns
        + l_existing_target_columns
        ]

    s_ml_csv_path = os.path.join(
        s_output_dir,
        f"{s_dataset_base_filename}_ml_ready.csv",
    )

    s_ml_excel_path = os.path.join(
        s_output_dir,
        f"{s_dataset_base_filename}_ml_ready.xlsx",
    )

    df_ml_ready.to_csv(
        s_ml_csv_path,
        index=False,
    )

    df_ml_ready.to_excel(
        s_ml_excel_path,
        index=False,
    )

    _print(
        "\n✅ ML-ready feature matrix exported."
    )

    _print(
        f"Rows:     {len(df_ml_ready)}"
    )

    _print(
        f"Features: {len(l_feature_columns)}"
    )

    _print(
        f"Targets:  {len(l_existing_target_columns)}"
    )

    _print(
        f"CSV:      {s_ml_csv_path}"
    )

    _print(
        f"Excel:    {s_ml_excel_path}"
    )

    _print(
        "\n✅ ML trace index exported."
    )

    _print(
        f"CSV:      {s_trace_csv_path}"
    )

    _print(
        f"Excel:    {s_trace_excel_path}"
    )

    return df_ml_ready


def get_cim_rdf_id_string(
        o_object,
):
    """
    Returns a clean CIM RDF ID string.

    PowerFactory may store cimRdfId as:
        ['_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx']

    This function returns:
        xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    """

    if o_object is None:
        return ""

    try:
        value = o_object.cimRdfId

        if isinstance(
                value,
                list,
        ):
            if not value:
                return ""

            value = value[0]

        return str(
            value,
        ).lstrip(
            "_",
        ).strip()

    except Exception:
        return ""


def load_switch_state_dataframe():
    """
    Loads Switch_state.csv.

    Expected format:
        ConfigID;switch_<rdf_id_1>;switch_<rdf_id_2>;...

    Switch values:
        1 = closed
        0 = open
    """

    if not b_enable_switch_state_scenarios:
        return pd.DataFrame([
            {
                "ConfigID": "live_grid_state",
            }
        ])

    if not os.path.exists(
            s_switch_state_file,
    ):
        raise RuntimeError(
            "❌ Switch-state file not found:\n"
            f"{s_switch_state_file}"
        )

    _print(
        "\n💾 Loading switch-state configuration file..."
    )

    df_switch_state = None

    for s_separator in [
        ";",
        ",",
        "\t",
    ]:
        try:
            df_candidate = pd.read_csv(
                s_switch_state_file,
                sep=s_separator,
            )

            if "ConfigID" in df_candidate.columns:
                df_switch_state = df_candidate
                break

        except Exception:
            pass

    if df_switch_state is None:
        raise RuntimeError(
            "❌ Could not read Switch_state.csv. "
            "Expected a file with a ConfigID column."
        )

    l_switch_columns = [
        s_column
        for s_column in df_switch_state.columns
        if str(s_column).startswith("switch_")
    ]

    if not l_switch_columns:
        raise RuntimeError(
            "❌ Switch_state.csv contains no switch_<rdf_id> columns."
        )

    for s_column in l_switch_columns:
        df_switch_state[s_column] = pd.to_numeric(
            df_switch_state[s_column],
            errors="coerce",
        ).fillna(
            1,
        ).astype(
            int,
        )

        df_switch_state[s_column] = df_switch_state[s_column].apply(
            lambda i_value: 1 if int(i_value) == 1 else 0
        )

    if i_max_switch_state_config_count is not None:
        df_switch_state = df_switch_state.head(
            int(
                i_max_switch_state_config_count,
            )
        )

    _print(
        f"✅ Switch-state configurations loaded: {len(df_switch_state)}"
    )

    _print(
        f"✅ Switch columns found: {len(l_switch_columns)}"
    )

    return df_switch_state


def build_cubicle_lookup_by_rdf_id():
    """
    Builds a lookup from cubicle CIM RDF ID to StaCubic object.
    """

    d_cubicle_by_rdf_id = {}

    try:
        l_all_cubicles = o_pf.GetCalcRelevantObjects(
            "*.StaCubic",
        ) or []

    except Exception:
        l_all_cubicles = []

    for o_cubicle in l_all_cubicles:
        s_rdf_id = get_cim_rdf_id_string(
            o_cubicle,
        )

        if not s_rdf_id:
            continue

        d_cubicle_by_rdf_id[s_rdf_id] = o_cubicle

    _print(
        f"✅ Cubicle RDF lookup created: {len(d_cubicle_by_rdf_id)} cubicles"
    )

    return d_cubicle_by_rdf_id


def get_first_cubicle_switch(
        o_cubicle,
):
    """
    Returns the first StaSwitch object inside a cubicle.
    """

    if o_cubicle is None:
        return None

    try:
        l_switches = o_cubicle.GetChildren(
            1,
            "*.StaSwitch",
        ) or []

        if l_switches:
            return l_switches[0]

    except Exception:
        pass

    return None


def store_original_switch_state(
        d_cubicle_by_rdf_id,
):
    """
    Stores original switch on_off values before switch-state scenarios.
    """

    d_original_switch_state = {}

    for s_rdf_id, o_cubicle in d_cubicle_by_rdf_id.items():
        o_switch = get_first_cubicle_switch(
            o_cubicle,
        )

        if o_switch is None:
            continue

        try:
            d_original_switch_state[s_rdf_id] = {
                "cubicle": o_cubicle,
                "switch": o_switch,
                "on_off": int(
                    o_switch.on_off,
                ),
            }

        except Exception:
            pass

    _print(
        f"✅ Original switch states stored: {len(d_original_switch_state)}"
    )

    return d_original_switch_state


def restore_original_switch_state(
        d_original_switch_state,
):
    """
    Restores original switch on_off values.
    """

    for _, d_switch_state in d_original_switch_state.items():
        o_switch = d_switch_state.get(
            "switch",
            None,
        )

        if o_switch is None:
            continue

        try:
            o_switch.on_off = int(
                d_switch_state.get(
                    "on_off",
                    1,
                )
            )

        except Exception:
            pass


def apply_switch_state_from_row(
        d_switch_state_row,
        d_cubicle_by_rdf_id,
        s_switch_state_config_id,
):
    """
    Applies one switch-state row to the live PowerFactory grid.

    Columns:
        switch_<cubicle_rdf_id>

    Values:
        1 = closed
        0 = open
    """

    i_switch_columns_seen = 0
    i_switches_applied = 0
    i_missing_cubicles = 0
    i_without_switch_object = 0
    i_closed_switches = 0
    i_open_switches = 0

    for s_column_name, value in d_switch_state_row.items():
        s_column_name = str(
            s_column_name,
        )

        if not s_column_name.startswith(
                "switch_",
        ):
            continue

        i_switch_columns_seen += 1

        s_rdf_id = s_column_name.split(
            "_",
            1,
        )[1].lstrip(
            "_",
        ).strip()

        o_cubicle = d_cubicle_by_rdf_id.get(
            s_rdf_id,
            None,
        )

        if o_cubicle is None:
            i_missing_cubicles += 1
            continue

        o_switch = get_first_cubicle_switch(
            o_cubicle,
        )

        if o_switch is None:
            i_without_switch_object += 1
            continue

        try:
            i_switch_value = int(
                value,
            )

        except Exception:
            i_switch_value = 1

        i_switch_value = 1 if i_switch_value == 1 else 0

        try:
            o_switch.on_off = i_switch_value
            i_switches_applied += 1

            if i_switch_value == 1:
                i_closed_switches += 1
            else:
                i_open_switches += 1

        except Exception:
            i_missing_cubicles += 1

    return {
        "switch_state_config_id": s_switch_state_config_id,
        "switch_state_switch_column_count": i_switch_columns_seen,
        "switch_state_applied_switch_count": i_switches_applied,
        "switch_state_missing_cubicle_count": i_missing_cubicles,
        "switch_state_without_switch_object_count": i_without_switch_object,
        "switch_state_closed_switch_count": i_closed_switches,
        "switch_state_open_switch_count": i_open_switches,
    }


def export_randomized_line_length_feature_matrix(
        o_project,
        o_grid,
):
    """
    Generates feature-matrix cases using three scenario layers:

        1. Switch-state topology scenarios from Switch_state.csv
        2. Line length/R/X randomization
        3. DG capacity randomization

    Scenario structure:
        for each switch state:
            export base_0000
            for each randomized scenario:
                restore original line and DG values
                re-apply switch state
                randomize line length/R/X
                randomize DG capacity
                export feature rows

    The original line, DG, and switch states are restored at the end.
    """

    _print(
        "\n⚙️ Creating randomized grid-scenario dataset with switch-state scenarios..."
    )

    # -------------------------------------------------------------------------
    # 1. Collect and store original line state
    # -------------------------------------------------------------------------
    l_randomizable_lines = get_randomizable_grid_lines(
        o_grid,
    )

    if not l_randomizable_lines:
        raise RuntimeError(
            "❌ No randomizable in-service lines found inside selected grid."
        )

    _print(
        f"✅ Randomizable lines found: {len(l_randomizable_lines)}"
    )

    d_original_line_state = store_original_line_length_impedance_state(
        l_randomizable_lines,
    )

    # -------------------------------------------------------------------------
    # 2. Collect and store original DG capacity state
    # -------------------------------------------------------------------------
    l_randomizable_dg = get_randomizable_grid_distributed_generators(
        o_grid,
    )

    _print(
        f"✅ Randomizable DGs found: {len(l_randomizable_dg)}"
    )

    d_original_dg_capacity_state = store_original_dg_capacity_state(
        l_randomizable_dg,
    )

    # -------------------------------------------------------------------------
    # 3. Load switch-state scenarios and store original switch state
    # -------------------------------------------------------------------------
    df_switch_state = load_switch_state_dataframe()

    d_cubicle_by_rdf_id = build_cubicle_lookup_by_rdf_id()

    d_original_switch_state = store_original_switch_state(
        d_cubicle_by_rdf_id,
    )

    l_switch_state_outserv_controlled_objects = get_switch_state_outserv_controlled_objects(
        o_grid,
    )

    d_original_component_outserv_state = get_original_outserv_states(
        l_switch_state_outserv_controlled_objects,
    )

    l_all_switch_state_component_outserv_log_rows = []

    # -------------------------------------------------------------------------
    # 4. Containers for combined results and logs
    # -------------------------------------------------------------------------
    l_all_scenario_dataframes = []
    l_all_line_randomization_log_rows = []
    l_all_dg_capacity_randomization_log_rows = []
    l_all_switch_state_application_log_rows = []

    try:
        # ---------------------------------------------------------------------
        # 5. Outer loop: switch-state topology scenarios
        # ---------------------------------------------------------------------
        for i_switch_state_index, d_switch_state_row in df_switch_state.iterrows():
            s_switch_state_config_id = str(
                d_switch_state_row.get(
                    "ConfigID",
                    f"switch_state_{i_switch_state_index + 1:04d}",
                )
            )

            s_switch_state_short_id = f"ss_{i_switch_state_index + 1:04d}"

            _print(
                "\n"
                + "=" * 80
            )

            _print(
                f"🔀 Applying switch-state configuration "
                f"{s_switch_state_short_id} | "
                f"ConfigID={s_switch_state_config_id}"
            )

            _print(
                "=" * 80
            )

            restore_original_outserv_states(
                d_original_component_outserv_state,
            )

            restore_original_switch_state(
                d_original_switch_state,
            )

            d_switch_application_summary = apply_switch_state_from_row(
                d_switch_state_row=d_switch_state_row.drop(
                    labels=[
                        "ConfigID",
                    ],
                    errors="ignore",
                ),
                d_cubicle_by_rdf_id=d_cubicle_by_rdf_id,
                s_switch_state_config_id=s_switch_state_config_id,
            )

            (
                d_outserv_summary,
                l_component_outserv_log_rows,
            ) = apply_outserv_for_components_behind_open_switches(
                l_component_objects=l_switch_state_outserv_controlled_objects,
                d_original_outserv_states=d_original_component_outserv_state,
                s_switch_state_short_id=s_switch_state_short_id,
                s_switch_state_config_id=s_switch_state_config_id,
                s_scenario_id="base_0000",
                b_collect_log_rows=True,
            )

            d_switch_application_summary.update(
                d_outserv_summary,
            )

            l_all_switch_state_component_outserv_log_rows.extend(
                l_component_outserv_log_rows,
            )

            l_all_switch_state_application_log_rows.append({
                "switch_state_row_index": i_switch_state_index + 1,
                "switch_state_short_id": s_switch_state_short_id,
                **d_switch_application_summary,
            })

            _print(
                f"✅ Switch state applied: "
                f"applied={d_switch_application_summary.get('switch_state_applied_switch_count', 0)} | "
                f"open={d_switch_application_summary.get('switch_state_open_switch_count', 0)} | "
                f"closed={d_switch_application_summary.get('switch_state_closed_switch_count', 0)}"
            )

            _print(
                "✅ Open-switch components forced out of service: "
                f"total={d_switch_application_summary.get('switch_state_forced_outserv_component_count', 0)} | "
                f"lines={d_switch_application_summary.get('switch_state_forced_outserv_line_count', 0)} | "
                f"DGs={d_switch_application_summary.get('switch_state_forced_outserv_dg_count', 0)} | "
                f"loads={d_switch_application_summary.get('switch_state_forced_outserv_load_count', 0)} | "
                f"transformers={d_switch_application_summary.get('switch_state_forced_outserv_transformer_count', 0)}"
            )

            if d_switch_application_summary.get(
                    "switch_state_missing_cubicle_count",
                    0,
            ) > 0:
                _print(
                    f"⚠️ Missing cubicles for {s_switch_state_short_id}: "
                    f"{d_switch_application_summary.get('switch_state_missing_cubicle_count', 0)}"
                )

            # -----------------------------------------------------------------
            # 5A. Base case for this switch state
            # -----------------------------------------------------------------
            if b_include_original_base_case_in_randomized_export:
                restore_original_line_length_impedance_state(
                    d_original_line_state,
                )

                restore_original_dg_capacity_state(
                    d_original_dg_capacity_state,
                )

                restore_original_outserv_states(
                    d_original_component_outserv_state,
                )

                restore_original_switch_state(
                    d_original_switch_state,
                )

                apply_switch_state_from_row(
                    d_switch_state_row=d_switch_state_row.drop(
                        labels=[
                            "ConfigID",
                        ],
                        errors="ignore",
                    ),
                    d_cubicle_by_rdf_id=d_cubicle_by_rdf_id,
                    s_switch_state_config_id=s_switch_state_config_id,
                )

                apply_outserv_for_components_behind_open_switches(
                    l_component_objects=l_switch_state_outserv_controlled_objects,
                    d_original_outserv_states=d_original_component_outserv_state,
                    s_switch_state_short_id=s_switch_state_short_id,
                    s_switch_state_config_id=s_switch_state_config_id,
                    s_scenario_id="base_0000",
                    b_collect_log_rows=False,
                )

                _print(
                    f"\n⚙️ Running {s_switch_state_short_id} | base_0000"
                )

                df_base = export_unified_corridor_feature_matrix(
                    o_project=o_project,
                    o_grid=o_grid,
                )

                df_base.insert(
                    0,
                    "switch_state_short_id",
                    s_switch_state_short_id,
                )

                df_base.insert(
                    1,
                    "switch_state_config_id",
                    s_switch_state_config_id,
                )

                df_base.insert(
                    2,
                    "switch_state_row_index",
                    i_switch_state_index + 1,
                )

                df_base.insert(
                    3,
                    "switch_state_open_switch_count",
                    d_switch_application_summary.get(
                        "switch_state_open_switch_count",
                        0,
                    ),
                )

                df_base.insert(
                    4,
                    "switch_state_closed_switch_count",
                    d_switch_application_summary.get(
                        "switch_state_closed_switch_count",
                        0,
                    ),
                )

                df_base.insert(
                    5,
                    "scenario_id",
                    "base_0000",
                )

                df_base.insert(
                    6,
                    "case_uid",
                    df_base["switch_state_short_id"].astype(str)
                    + "_"
                    + df_base["scenario_id"].astype(str)
                    + "_case_"
                    + df_base["case_id"].astype(str).str.zfill(3),
                )

                df_base.insert(
                    7,
                    "line_length_random_seed",
                    "",
                )

                df_base.insert(
                    8,
                    "line_length_scale_min",
                    1.0,
                )

                df_base.insert(
                    9,
                    "line_length_scale_max",
                    1.0,
                )

                df_base.insert(
                    10,
                    "dg_capacity_random_seed",
                    "",
                )

                df_base.insert(
                    11,
                    "dg_capacity_scale_min",
                    1.0,
                )

                df_base.insert(
                    12,
                    "dg_capacity_scale_max",
                    1.0,
                )

                l_all_scenario_dataframes.append(
                    df_base,
                )

            # -----------------------------------------------------------------
            # 5B. Randomized line/DG scenarios for this switch state
            # -----------------------------------------------------------------
            for i_scenario in range(
                    1,
                    i_randomized_scenario_count + 1,
            ):
                s_scenario_id = f"rand_{i_scenario:04d}"

                i_line_random_seed = (
                        i_random_seed_base
                        + i_scenario
                        + (
                                i_switch_state_index
                                * i_randomized_scenario_count
                        )
                )

                i_dg_capacity_random_seed = (
                        i_line_random_seed
                        + i_dg_capacity_random_seed_offset
                )

                _print(
                    f"\n⚙️ Running {s_switch_state_short_id} | "
                    f"{s_scenario_id} "
                    f"| line_seed={i_line_random_seed} "
                    f"| line_scale=[{f_line_length_scale_min}, {f_line_length_scale_max}] "
                    f"| dg_seed={i_dg_capacity_random_seed if b_enable_dg_capacity_randomization else 'OFF'} "
                    f"| dg_scale=[{f_dg_capacity_scale_min}, {f_dg_capacity_scale_max}]"
                )

                # -------------------------------------------------------------
                # Every randomized scenario starts from the same base line/DG
                # values, then the same switch state is re-applied.
                # -------------------------------------------------------------
                restore_original_line_length_impedance_state(
                    d_original_line_state,
                )

                restore_original_dg_capacity_state(
                    d_original_dg_capacity_state,
                )

                restore_original_outserv_states(
                    d_original_component_outserv_state,
                )

                restore_original_switch_state(
                    d_original_switch_state,
                )

                apply_switch_state_from_row(
                    d_switch_state_row=d_switch_state_row.drop(
                        labels=[
                            "ConfigID",
                        ],
                        errors="ignore",
                    ),
                    d_cubicle_by_rdf_id=d_cubicle_by_rdf_id,
                    s_switch_state_config_id=s_switch_state_config_id,
                )

                apply_outserv_for_components_behind_open_switches(
                    l_component_objects=l_switch_state_outserv_controlled_objects,
                    d_original_outserv_states=d_original_component_outserv_state,
                    s_switch_state_short_id=s_switch_state_short_id,
                    s_switch_state_config_id=s_switch_state_config_id,
                    s_scenario_id=s_scenario_id,
                    b_collect_log_rows=False,
                )

                # -------------------------------------------------------------
                # Apply line length/R/X randomization.
                # -------------------------------------------------------------
                l_scenario_line_randomization_log_rows = apply_random_line_length_scenario(
                    d_original_state=d_original_line_state,
                    s_scenario_id=f"{s_switch_state_short_id}_{s_scenario_id}",
                    i_random_seed=i_line_random_seed,
                    f_scale_min=f_line_length_scale_min,
                    f_scale_max=f_line_length_scale_max,
                )

                for d_log_row in l_scenario_line_randomization_log_rows:
                    d_log_row["switch_state_short_id"] = s_switch_state_short_id
                    d_log_row["switch_state_config_id"] = s_switch_state_config_id
                    d_log_row["switch_state_row_index"] = i_switch_state_index + 1

                l_all_line_randomization_log_rows.extend(
                    l_scenario_line_randomization_log_rows,
                )

                # -------------------------------------------------------------
                # Apply DG capacity randomization.
                # -------------------------------------------------------------
                if b_enable_dg_capacity_randomization:
                    l_scenario_dg_capacity_log_rows = apply_random_dg_capacity_scenario(
                        d_original_state=d_original_dg_capacity_state,
                        s_scenario_id=f"{s_switch_state_short_id}_{s_scenario_id}",
                        i_random_seed=i_dg_capacity_random_seed,
                        f_scale_min=f_dg_capacity_scale_min,
                        f_scale_max=f_dg_capacity_scale_max,
                    )

                    for d_log_row in l_scenario_dg_capacity_log_rows:
                        d_log_row["switch_state_short_id"] = s_switch_state_short_id
                        d_log_row["switch_state_config_id"] = s_switch_state_config_id
                        d_log_row["switch_state_row_index"] = i_switch_state_index + 1

                    l_all_dg_capacity_randomization_log_rows.extend(
                        l_scenario_dg_capacity_log_rows,
                    )

                # -------------------------------------------------------------
                # Run feature extraction after topology and randomization.
                # -------------------------------------------------------------
                df_scenario = export_unified_corridor_feature_matrix(
                    o_project=o_project,
                    o_grid=o_grid,
                )

                df_scenario.insert(
                    0,
                    "switch_state_short_id",
                    s_switch_state_short_id,
                )

                df_scenario.insert(
                    1,
                    "switch_state_config_id",
                    s_switch_state_config_id,
                )

                df_scenario.insert(
                    2,
                    "switch_state_row_index",
                    i_switch_state_index + 1,
                )

                df_scenario.insert(
                    3,
                    "switch_state_open_switch_count",
                    d_switch_application_summary.get(
                        "switch_state_open_switch_count",
                        0,
                    ),
                )

                df_scenario.insert(
                    4,
                    "switch_state_closed_switch_count",
                    d_switch_application_summary.get(
                        "switch_state_closed_switch_count",
                        0,
                    ),
                )

                df_scenario.insert(
                    5,
                    "scenario_id",
                    s_scenario_id,
                )

                df_scenario.insert(
                    6,
                    "case_uid",
                    df_scenario["switch_state_short_id"].astype(str)
                    + "_"
                    + df_scenario["scenario_id"].astype(str)
                    + "_case_"
                    + df_scenario["case_id"].astype(str).str.zfill(3),
                )

                df_scenario.insert(
                    7,
                    "line_length_random_seed",
                    i_line_random_seed,
                )

                df_scenario.insert(
                    8,
                    "line_length_scale_min",
                    f_line_length_scale_min,
                )

                df_scenario.insert(
                    9,
                    "line_length_scale_max",
                    f_line_length_scale_max,
                )

                df_scenario.insert(
                    10,
                    "dg_capacity_random_seed",
                    i_dg_capacity_random_seed
                    if b_enable_dg_capacity_randomization
                    else "",
                )

                df_scenario.insert(
                    11,
                    "dg_capacity_scale_min",
                    f_dg_capacity_scale_min
                    if b_enable_dg_capacity_randomization
                    else 1.0,
                )

                df_scenario.insert(
                    12,
                    "dg_capacity_scale_max",
                    f_dg_capacity_scale_max
                    if b_enable_dg_capacity_randomization
                    else 1.0,
                )

                l_all_scenario_dataframes.append(
                    df_scenario,
                )


    finally:

        restore_original_line_length_impedance_state(

            d_original_line_state,

        )

        restore_original_dg_capacity_state(

            d_original_dg_capacity_state,

        )

        restore_original_outserv_states(

            d_original_component_outserv_state,

        )

        restore_original_switch_state(

            d_original_switch_state,

        )

    # -------------------------------------------------------------------------
    # 6. Combine scenario dataframes
    # -------------------------------------------------------------------------
    if not l_all_scenario_dataframes:
        raise RuntimeError(
            "❌ No scenario dataframes were created. "
            "Enable base case export or set randomized scenario count > 0."
        )

    df_randomized_all = pd.concat(
        l_all_scenario_dataframes,
        ignore_index=True,
    )

    s_output_dir = get_feature_matrix_output_directory(
        o_project,
        o_grid,
    )

    os.makedirs(
        s_output_dir,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 7. Export combined randomized feature matrix
    # -------------------------------------------------------------------------
    s_combined_csv_path = os.path.join(
        s_output_dir,
        "case_feature_matrix_randomized_grid_scenarios.csv",
    )

    s_combined_excel_path = os.path.join(
        s_output_dir,
        "case_feature_matrix_randomized_grid_scenarios.xlsx",
    )

    df_randomized_all.to_csv(
        s_combined_csv_path,
        index=False,
    )

    df_randomized_all.to_excel(
        s_combined_excel_path,
        index=False,
    )

    df_ml_ready = export_ml_ready_feature_matrix_from_dataframe(
        df_feature_matrix=df_randomized_all,
        s_output_dir=s_output_dir,
        s_dataset_base_filename="case_feature_matrix_randomized_grid_scenarios",
    )

    _print(
        "\n✅ Randomized grid-scenario feature matrix exported."
    )

    _print(
        f"Rows:  {len(df_randomized_all)}"
    )

    _print(
        f"CSV:   {s_combined_csv_path}"
    )

    _print(
        f"Excel: {s_combined_excel_path}"
    )

    # -------------------------------------------------------------------------
    # 8. Export line length/R/X randomization log
    # -------------------------------------------------------------------------
    df_line_randomization_log = pd.DataFrame(
        l_all_line_randomization_log_rows,
    )

    s_line_log_csv_path = os.path.join(
        s_output_dir,
        "line_length_randomization_log.csv",
    )

    s_line_log_excel_path = os.path.join(
        s_output_dir,
        "line_length_randomization_log.xlsx",
    )

    df_line_randomization_log.to_csv(
        s_line_log_csv_path,
        index=False,
    )

    df_line_randomization_log.to_excel(
        s_line_log_excel_path,
        index=False,
    )

    _print(
        "\n✅ Line length/R/X randomization log exported."
    )

    _print(
        f"CSV:   {s_line_log_csv_path}"
    )

    _print(
        f"Excel: {s_line_log_excel_path}"
    )

    # -------------------------------------------------------------------------
    # 9. Export DG capacity randomization log
    # -------------------------------------------------------------------------
    df_dg_capacity_randomization_log = pd.DataFrame(
        l_all_dg_capacity_randomization_log_rows,
    )

    s_dg_log_csv_path = os.path.join(
        s_output_dir,
        "dg_capacity_randomization_log.csv",
    )

    s_dg_log_excel_path = os.path.join(
        s_output_dir,
        "dg_capacity_randomization_log.xlsx",
    )

    df_dg_capacity_randomization_log.to_csv(
        s_dg_log_csv_path,
        index=False,
    )

    df_dg_capacity_randomization_log.to_excel(
        s_dg_log_excel_path,
        index=False,
    )

    _print(
        "\n✅ DG capacity randomization log exported."
    )

    _print(
        f"CSV:   {s_dg_log_csv_path}"
    )

    _print(
        f"Excel: {s_dg_log_excel_path}"
    )

    # -------------------------------------------------------------------------
    # 10. Export switch-state application log
    # -------------------------------------------------------------------------
    df_switch_state_application_log = pd.DataFrame(
        l_all_switch_state_application_log_rows,
    )

    s_switch_log_csv_path = os.path.join(
        s_output_dir,
        "switch_state_application_log.csv",
    )

    s_switch_log_excel_path = os.path.join(
        s_output_dir,
        "switch_state_application_log.xlsx",
    )

    df_switch_state_application_log.to_csv(
        s_switch_log_csv_path,
        index=False,
    )

    df_switch_state_application_log.to_excel(
        s_switch_log_excel_path,
        index=False,
    )

    _print(
        "\n✅ Switch-state application log exported."
    )

    _print(
        f"CSV:   {s_switch_log_csv_path}"
    )

    _print(
        f"Excel: {s_switch_log_excel_path}"
    )

    # -------------------------------------------------------------------------
    # 11. Export switch-state component outserv derivation log
    # -------------------------------------------------------------------------
    if l_all_switch_state_component_outserv_log_rows:
        df_component_outserv_log = pd.DataFrame(
            l_all_switch_state_component_outserv_log_rows,
        )

        s_component_outserv_log_csv_path = os.path.join(
            s_output_dir,
            "switch_state_component_outserv_log.csv",
        )

        s_component_outserv_log_excel_path = os.path.join(
            s_output_dir,
            "switch_state_component_outserv_log.xlsx",
        )

        df_component_outserv_log.to_csv(
            s_component_outserv_log_csv_path,
            index=False,
        )

        df_component_outserv_log.to_excel(
            s_component_outserv_log_excel_path,
            index=False,
        )

        _print(
            "\n✅ Switch-state component outserv derivation log exported."
        )

        _print(
            f"CSV:   {s_component_outserv_log_csv_path}"
        )

        _print(
            f"Excel: {s_component_outserv_log_excel_path}"
        )

    return df_randomized_all


#Check only
def validate_reach_correction_addition(df_case_feature_matrix):
    """
    Validates that:

        target_zone1 = zone1_base + zone1_infeed_correction
        target_zone2 = zone2_base + zone2_infeed_correction

    within a small rounding tolerance.
    """

    f_tolerance = 1e-5

    l_required_columns = [
        "case_id",
        "protected_corridor_id",

        "base_zone1_r_reach_ohm",
        "base_zone1_x_reach_ohm",
        "zone1_infeed_correction_r_ohm",
        "zone1_infeed_correction_x_ohm",
        "target_zone1_r_reach_ohm",
        "target_zone1_x_reach_ohm",

        "base_zone2_r_reach_ohm",
        "base_zone2_x_reach_ohm",
        "zone2_infeed_correction_r_ohm",
        "zone2_infeed_correction_x_ohm",
        "target_zone2_r_reach_ohm",
        "target_zone2_x_reach_ohm",
    ]

    for s_column in l_required_columns:
        if s_column not in df_case_feature_matrix.columns:
            _print(
                f"⚠️ Reach validation skipped. Missing column: {s_column}"
            )
            return

    df_check = df_case_feature_matrix.copy()

    l_numeric_columns = [
        "base_zone1_r_reach_ohm",
        "base_zone1_x_reach_ohm",
        "zone1_infeed_correction_r_ohm",
        "zone1_infeed_correction_x_ohm",
        "target_zone1_r_reach_ohm",
        "target_zone1_x_reach_ohm",

        "base_zone2_r_reach_ohm",
        "base_zone2_x_reach_ohm",
        "zone2_infeed_correction_r_ohm",
        "zone2_infeed_correction_x_ohm",
        "target_zone2_r_reach_ohm",
        "target_zone2_x_reach_ohm",
    ]

    for s_column in l_numeric_columns:
        df_check[s_column] = pd.to_numeric(
            df_check[s_column],
            errors="coerce",
        ).fillna(
            0.0,
        )

    df_check["zone1_r_error"] = (
            df_check["target_zone1_r_reach_ohm"]
            - (
                    df_check["base_zone1_r_reach_ohm"]
                    + df_check["zone1_infeed_correction_r_ohm"]
            )
    )

    df_check["zone1_x_error"] = (
            df_check["target_zone1_x_reach_ohm"]
            - (
                    df_check["base_zone1_x_reach_ohm"]
                    + df_check["zone1_infeed_correction_x_ohm"]
            )
    )

    df_check["zone2_r_error"] = (
            df_check["target_zone2_r_reach_ohm"]
            - (
                    df_check["base_zone2_r_reach_ohm"]
                    + df_check["zone2_infeed_correction_r_ohm"]
            )
    )

    df_check["zone2_x_error"] = (
            df_check["target_zone2_x_reach_ohm"]
            - (
                    df_check["base_zone2_x_reach_ohm"]
                    + df_check["zone2_infeed_correction_x_ohm"]
            )
    )

    df_failed = df_check[
        (
                df_check["zone1_r_error"].abs() > f_tolerance
        )
        | (
                df_check["zone1_x_error"].abs() > f_tolerance
        )
        | (
                df_check["zone2_r_error"].abs() > f_tolerance
        )
        | (
                df_check["zone2_x_error"].abs() > f_tolerance
        )
        ]

    if df_failed.empty:
        _print(
            "\n✅ Reach correction validation passed: "
            "target = base + correction for Zone 1 and Zone 2."
        )
        return

    _print(
        "⚠️ Reach correction validation failed for these cases:"
    )

    for _, d_row in df_failed.iterrows():
        _print(
            f"   {d_row.get('case_id', '')} | "
            f"{d_row.get('protected_corridor_id', '')} | "
            f"Z1_error=({d_row['zone1_r_error']:.8f}, "
            f"{d_row['zone1_x_error']:.8f}) | "
            f"Z2_error=({d_row['zone2_r_error']:.8f}, "
            f"{d_row['zone2_x_error']:.8f})"
        )


# =============================================================================
# Turbine Location Debug Functions
# =============================================================================

def find_known_corridor(
        l_protected_corridors,
        s_relay_busbar_name,
        s_subsequent_busbar_name,
        s_protected_corridor_id,
):
    """
    Finds one known directional corridor for debug.

    Example:
        Terminal_HV_B -> Terminal_HV_C | Line_HV_B-C
    """

    for d_corridor in l_protected_corridors:
        s_relay_busbar = _get_safe_name(
            d_corridor.get("relay_busbar"),
        )

        s_subsequent_busbar = _get_safe_name(
            d_corridor.get("subsequent_busbar"),
        )

        s_edge_id = d_corridor.get(
            "protected_corridor_id",
            "",
        )

        if (
                s_relay_busbar == s_relay_busbar_name
                and s_subsequent_busbar == s_subsequent_busbar_name
                and s_edge_id == s_protected_corridor_id
        ):
            return d_corridor

    return None


def get_terminal_for_distributed_generator(o_dg):
    """
    Returns terminal where a distributed generator is connected.
    """

    o_cubicle = _get_attribute_object(
        o_dg,
        "bus1",
    )

    return _get_terminal_from_cubicle(
        o_cubicle,
    )


def get_branch_line_names(d_branch):
    return " -> ".join([
        _get_safe_name(o_line)
        for o_line in d_branch.get("Branch Lines", [])
    ])


def get_corridor_line_names(d_corridor):
    return " -> ".join([
        _get_safe_name(o_line)
        for o_line in d_corridor.get("line_sections", [])
    ])


def get_fault_location_at_line_half_length(o_line):
    """
    Returns 50 percent fault location on a single line section.
    """

    f_line_length_km = get_line_length_value(
        o_line,
    )

    return {
        "selected_fault_line": _get_safe_name(o_line),
        "selected_fault_location_rule": "50_percent_of_selected_line",
        "selected_fault_location_km": 0.5 * f_line_length_km,
        "selected_fault_percent": 50.0,
    }


def get_fault_location_for_last_branch_section(d_branch):
    """
    For downstream-branch junction DGs, use the 50 percent point of the last
    branch line section.

    This follows the idea from _zone_calculation.py:
    if the turbine is on a junction-node multi-section branch, fault placement
    is related to the last section of that branch.
    """

    l_branch_lines = d_branch.get("Branch Lines", [])

    if not l_branch_lines:
        return {
            "selected_fault_line": "",
            "selected_fault_location_rule": "no_downstream_branch_line",
            "selected_fault_location_km": 0.0,
            "selected_fault_percent": 0.0,
        }

    o_last_line = l_branch_lines[-1]

    d_fault = get_fault_location_at_line_half_length(
        o_last_line,
    )

    d_fault["selected_fault_location_rule"] = "50_percent_of_last_downstream_branch_section"

    return d_fault


def get_fault_location_for_first_branch_section(d_branch):
    """
    For a DG located exactly at the remote busbar, use the 50 percent point
    of the first downstream branch section.

    Example for Terminal_HV_B -> Terminal_HV_C:
        RemoteBusbarDG = Wind_HV_C
        Downstream branch starts with Line_HV_C-D_1
        Fault = 50 percent of Line_HV_C-D_1
    """

    l_branch_lines = d_branch.get("Branch Lines", [])

    if not l_branch_lines:
        return {
            "selected_fault_line": "",
            "selected_fault_location_rule": "no_downstream_branch_line_for_subsequent_busbar_dg",
            "selected_fault_location_km": 0.0,
            "selected_fault_percent": 0.0,
        }

    o_first_line = l_branch_lines[0]

    d_fault = get_fault_location_at_line_half_length(
        o_first_line,
    )

    d_fault["selected_fault_location_rule"] = "50_percent_of_first_downstream_branch_section"

    return d_fault


def get_fault_location_for_protected_corridor_last_section(d_corridor):
    """
    For DGs located on a protected multi-section corridor, use 50 percent of
    the last section of the protected corridor for the first debug version.
    """

    l_line_sections = d_corridor.get("line_sections", [])

    if not l_line_sections:
        return {
            "selected_fault_line": "",
            "selected_fault_location_rule": "no_protected_corridor_line",
            "selected_fault_location_km": 0.0,
            "selected_fault_percent": 0.0,
        }

    o_last_line = l_line_sections[-1]

    d_fault = get_fault_location_at_line_half_length(
        o_last_line,
    )

    d_fault["selected_fault_location_rule"] = "50_percent_of_last_protected_corridor_section"

    return d_fault


def calculate_turbine_position_fraction_on_terminal_path(o_turbine_terminal, l_path_terminals):
    """
    Returns approximate turbine terminal position along a terminal path.

    0.0 = first terminal
    1.0 = last terminal
    """

    if not l_path_terminals:
        return 0.0

    if len(l_path_terminals) == 1:
        return 0.0

    for i_index, o_terminal in enumerate(l_path_terminals):
        if o_terminal == o_turbine_terminal:
            return round(
                i_index / (len(l_path_terminals) - 1),
                3,
            )

    return 0.0


def get_zone_location_candidate_for_dg_category(s_dg_context_category):
    """
    First-pass zone candidate classification.

    This is debug only, not final correction logic.
    """

    if s_dg_context_category == "relay_busbar":
        return "relay_side_not_downstream_infeed"

    if s_dg_context_category == "subsequent_busbar":
        return "subsequent_busbar_boundary_candidate"

    if s_dg_context_category == "protected_corridor":
        return "inside_protected_corridor_candidate"

    if s_dg_context_category == "downstream_branch":
        return "zone2_downstream_branch_candidate"

    if s_dg_context_category == "remote_busbar":
        return "far_end_or_later_network_candidate"

    return "unknown"


# =============================================================================
# Known Corridor Short-Circuit Infeed Debug From Turbine Rows
# =============================================================================

def assign_fault_from_turbine_debug_row(
        o_shc,
        o_fault_line,
        f_fault_location_km,
        f_fault_percent,
):
    """
    Assigns the short-circuit fault using values from the turbine debug row.
    """

    if o_fault_line is None:
        return False

    try:
        o_shc.shcobj = o_fault_line
        o_shc.faultloc = float(f_fault_location_km)
        o_shc.ppro = float(f_fault_percent)

        return True

    except Exception as o_error:
        _print(f"⚠️ Could not assign fault from turbine debug row: {o_error}")
        return False


# =============================================================================
# Zone-Based Infeed Correction Debug Functions
# =============================================================================

def get_line_impedance_fraction(o_line, f_fraction):
    """
    Returns R/X impedance of a line section multiplied by a fraction.
    """

    f_r, f_x = _get_line_impedance(
        o_line,
    )

    f_fraction = max(
        0.0,
        min(
            1.0,
            float(f_fraction or 0.0),
        ),
    )

    return (
        f_r * f_fraction,
        f_x * f_fraction,
    )


def get_impedance_from_ordered_lines_to_fault(
        l_ordered_lines,
        s_selected_fault_line,
        f_selected_fault_percent,
):
    """
    Sums impedance from the beginning of an ordered line list up to the fault.

    Full impedance is used for all line sections before the selected fault line.
    Partial impedance is used for the selected fault line.
    """

    f_total_r = 0.0
    f_total_x = 0.0

    f_fault_fraction = float(
        f_selected_fault_percent or 0.0
    ) / 100.0

    l_used_line_names = []

    for o_line in l_ordered_lines:
        s_line_name = _get_safe_name(
            o_line,
        )

        if s_line_name == s_selected_fault_line:
            f_r, f_x = get_line_impedance_fraction(
                o_line,
                f_fault_fraction,
            )

            f_total_r += f_r
            f_total_x += f_x
            l_used_line_names.append(
                f"{s_line_name}({round(f_fault_fraction, 3)})"
            )

            break

        f_r, f_x = get_line_impedance_fraction(
            o_line,
            1.0,
        )

        f_total_r += f_r
        f_total_x += f_x
        l_used_line_names.append(
            s_line_name,
        )

    return {
        "impedance_to_fault_r_ohm": round(f_total_r, 3),
        "impedance_to_fault_x_ohm": round(f_total_x, 3),
        "impedance_to_fault_line_path": " -> ".join(l_used_line_names),
    }


def get_impedance_from_branch_terminal_to_fault(
        d_shortest_branch,
        s_connected_terminal_name,
        s_selected_fault_line,
        f_selected_fault_percent,
):
    """
    Calculates impedance from a turbine terminal on the downstream branch
    to the selected fault point.

    Branch terminal order:
        Terminal_HV_C -> Terminal_HV_C-D_1 -> ... -> Terminal_HV_D

    If turbine is at Terminal_HV_C-D_2, impedance starts after that terminal:
        Line_HV_C-D_3 -> ... -> fault line
    """

    l_branch_terminals = d_shortest_branch.get(
        "Branch Terminals",
        [],
    )

    l_branch_lines = d_shortest_branch.get(
        "Branch Lines",
        [],
    )

    i_start_terminal_index = None

    for i_index, o_terminal in enumerate(l_branch_terminals):
        if _get_safe_name(o_terminal) == s_connected_terminal_name:
            i_start_terminal_index = i_index
            break

    if i_start_terminal_index is None:
        return {
            "impedance_to_fault_r_ohm": 0.0,
            "impedance_to_fault_x_ohm": 0.0,
            "impedance_to_fault_line_path": "",
        }

    l_lines_from_terminal_to_fault = l_branch_lines[
                                     i_start_terminal_index:
                                     ]

    return get_impedance_from_ordered_lines_to_fault(
        l_lines_from_terminal_to_fault,
        s_selected_fault_line,
        f_selected_fault_percent,
    )


def get_impedance_from_corridor_terminal_to_fault(
        d_corridor,
        s_connected_terminal_name,
        s_selected_fault_line,
        f_selected_fault_percent,
):
    """
    Calculates impedance from a turbine terminal on the protected corridor
    to the selected fault point.

    Used for protected-corridor DG cases.
    """

    l_corridor_terminals = get_ordered_terminals_along_protected_corridor(
        d_corridor,
    )

    l_corridor_lines = d_corridor.get(
        "line_sections",
        [],
    )

    i_start_terminal_index = None

    for i_index, o_terminal in enumerate(l_corridor_terminals):
        if _get_safe_name(o_terminal) == s_connected_terminal_name:
            i_start_terminal_index = i_index
            break

    if i_start_terminal_index is None:
        return {
            "impedance_to_fault_r_ohm": 0.0,
            "impedance_to_fault_x_ohm": 0.0,
            "impedance_to_fault_line_path": "",
        }

    l_lines_from_terminal_to_fault = l_corridor_lines[
                                     i_start_terminal_index:
                                     ]

    return get_impedance_from_ordered_lines_to_fault(
        l_lines_from_terminal_to_fault,
        s_selected_fault_line,
        f_selected_fault_percent,
    )


def get_reach_based_zone_correction_bucket(
        s_dg_context_category,
        f_turbine_position_x_ohm,
        d_reach_reference,
):
    """
    Reach-based bucket classification.

    Uses X reach first, matching the branch-selection/reactance style.
    """

    if s_dg_context_category == "relay_busbar":
        return "skip_relay_side_dg"

    if s_dg_context_category == "remote_busbar":
        return "skip_remote_busbar_dg"

    f_position_x = float(
        f_turbine_position_x_ohm or 0.0
    )

    f_zone1_x = float(
        d_reach_reference.get("zone1_x_reach_ohm", 0.0) or 0.0
    )

    f_intended_zone2_x = float(
        d_reach_reference.get("intended_zone2_x_reach_ohm", 0.0) or 0.0
    )

    if f_position_x <= f_zone1_x:
        return "zone1_candidate"

    if f_position_x <= f_intended_zone2_x:
        return "zone2_candidate"

    return "outside_intended_zone2"


def get_turbine_position_impedance_from_relay(
        d_known_corridor,
        d_shortest_branch,
        s_dg_context_category,
        s_connected_terminal_name,
):
    """
    Calculates approximate relay-to-turbine position impedance.

    This is used only for bucket classification:
        Zone 1 / Zone 2 / outside Zone 2.
    """

    if s_dg_context_category == "relay_busbar":
        return {
            "turbine_position_r_ohm": 0.0,
            "turbine_position_x_ohm": 0.0,
            "turbine_position_path": "",
        }

    f_total_r = 0.0
    f_total_x = 0.0
    l_path_names = []

    for o_line in d_known_corridor.get("line_sections", []):
        f_r, f_x = _get_line_impedance(
            o_line,
        )

        f_total_r += f_r
        f_total_x += f_x
        l_path_names.append(
            _get_safe_name(o_line),
        )

    if s_dg_context_category == "subsequent_busbar":
        return {
            "turbine_position_r_ohm": round(f_total_r, 3),
            "turbine_position_x_ohm": round(f_total_x, 3),
            "turbine_position_path": " -> ".join(l_path_names),
        }

    if s_dg_context_category == "protected_corridor":
        l_corridor_terminals = get_ordered_terminals_along_protected_corridor(
            d_known_corridor,
        )

        f_total_r = 0.0
        f_total_x = 0.0
        l_path_names = []

        for i_index, o_line in enumerate(d_known_corridor.get("line_sections", [])):
            if i_index + 1 >= len(l_corridor_terminals):
                break

            f_r, f_x = _get_line_impedance(
                o_line,
            )

            f_total_r += f_r
            f_total_x += f_x
            l_path_names.append(
                _get_safe_name(o_line),
            )

            if _get_safe_name(l_corridor_terminals[i_index + 1]) == s_connected_terminal_name:
                break

        return {
            "turbine_position_r_ohm": round(f_total_r, 6),
            "turbine_position_x_ohm": round(f_total_x, 6),
            "turbine_position_path": " -> ".join(l_path_names),
        }

    if s_dg_context_category in [
        "downstream_branch",
        "remote_busbar",
    ]:
        for o_line in d_shortest_branch.get("Branch Lines", []):
            o_terminal_1, o_terminal_2, _, _ = get_line_terminals(
                o_line,
            )

            f_r, f_x = _get_line_impedance(
                o_line,
            )

            f_total_r += f_r
            f_total_x += f_x
            l_path_names.append(
                _get_safe_name(o_line),
            )

            if (
                    _get_safe_name(o_terminal_1) == s_connected_terminal_name
                    or _get_safe_name(o_terminal_2) == s_connected_terminal_name
            ):
                break

        return {
            "turbine_position_r_ohm": round(f_total_r, 3),
            "turbine_position_x_ohm": round(f_total_x, 3),
            "turbine_position_path": " -> ".join(l_path_names),
        }

    return {
        "turbine_position_r_ohm": 0.0,
        "turbine_position_x_ohm": 0.0,
        "turbine_position_path": "",
    }


# =============================================================================
# All Corridors Short-Circuit Infeed Debug
# =============================================================================

def find_corridor_from_names(
        l_protected_corridors,
        s_relay_busbar,
        s_subsequent_busbar,
        s_protected_corridor_id,
):
    """
    Finds a protected corridor using relay busbar, remote busbar, and edge id.
    """

    for d_corridor in l_protected_corridors:
        if _get_safe_name(d_corridor.get("relay_busbar")) != s_relay_busbar:
            continue

        if _get_safe_name(d_corridor.get("subsequent_busbar")) != s_subsequent_busbar:
            continue

        if str(d_corridor.get("protected_corridor_id", "")) != str(s_protected_corridor_id):
            continue

        return d_corridor

    return None


def find_fault_line_object_for_sc_row(
        d_corridor,
        d_shortest_branch,
        s_selected_fault_line,
):
    """
    Finds selected fault line object from corridor or downstream branch.
    """

    l_candidate_lines = []

    l_candidate_lines.extend(
        d_corridor.get("line_sections", [])
    )

    l_candidate_lines.extend(
        d_shortest_branch.get("Branch Lines", [])
    )

    for o_line in l_candidate_lines:
        if _get_safe_name(o_line) == s_selected_fault_line:
            return o_line

    return None


def assign_short_circuit_fault_from_debug_row(
        o_shc,
        o_fault_line,
        f_selected_fault_location_km,
        f_selected_fault_percent,
):
    """
    Assigns short-circuit fault from turbine-location debug row.
    """

    if o_shc is None or o_fault_line is None:
        return False

    try:
        o_shc.shcobj = o_fault_line
        o_shc.faultloc = float(
            f_selected_fault_location_km or 0.0
        )
        o_shc.ppro = float(
            f_selected_fault_percent or 0.0
        )

        return True

    except Exception as o_error:
        _print(f"⚠️ Could not assign SC fault from debug row: {o_error}")
        return False


# =============================================================================
# All Corridors Zone-Based Correction Debug
# =============================================================================

def get_impedance_to_fault_for_all_corridors_row(
        d_corridor,
        d_shortest_branch,
        s_context,
        s_connected_terminal,
        s_selected_fault_line,
        f_selected_fault_percent,
):
    """
    Calculates impedance from the DG terminal to the selected fault point.

    This follows the same correction idea as the known-corridor debug:
        correction = impedance_to_fault * DG/relay ratio
    """

    if s_context == "relay_busbar":
        return {
            "impedance_to_fault_r_ohm": 0.0,
            "impedance_to_fault_x_ohm": 0.0,
            "impedance_to_fault_line_path": "",
        }

    if s_context in [
        "subsequent_busbar",
        "downstream_branch",
        "remote_busbar",
    ]:
        return get_impedance_from_branch_terminal_to_fault(
            d_shortest_branch,
            s_connected_terminal,
            s_selected_fault_line,
            f_selected_fault_percent,
        )

    if s_context == "protected_corridor":
        return get_impedance_from_corridor_terminal_to_fault(
            d_corridor,
            s_connected_terminal,
            s_selected_fault_line,
            f_selected_fault_percent,
        )

    return {
        "impedance_to_fault_r_ohm": 0.0,
        "impedance_to_fault_x_ohm": 0.0,
        "impedance_to_fault_line_path": "",
    }


# =============================================================================
# Main
# =============================================================================

def main():
    global o_pf

    _print("\n" + "=" * 80)
    _print(
        "v0.1 - Unified corridor-based feature matrix "
        "with switch-state topology scenarios, randomized grid-scenario and ML-ready export"
    )
    _print("=" * 80)

    o_pf = connect_to_powerfactory(
        b_enable_debug=b_enable_debug,
    )

    prepare_working_directory(
        s_dir_codingbase,
    )

    o_project = activate_powerfactory_project(
        o_pf,
        s_projectname,
    )

    o_grid_110 = get_target_grid(
        o_project,
        s_grid_name,
    )

    if b_enable_line_length_randomization:
        df_case_feature_matrix = export_randomized_line_length_feature_matrix(
            o_project,
            o_grid_110,
        )
    else:
        df_case_feature_matrix = export_unified_corridor_feature_matrix(
            o_project,
            o_grid_110,
        )

    _print("\n✅ Unified corridor-based feature matrix generation completed.")

    return df_case_feature_matrix


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    main()
