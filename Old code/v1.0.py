# =============================================================================
# Unified corridor-based feature matrix generation
# =============================================================================

import os
import sys
import random
import logging
from pathlib import Path
from collections import defaultdict
import pandas as pd

# =============================================================================
# Configuration & Paths
# =============================================================================
# SCRIPT_DIR is the folder where distance_protection_dataset_generator.py lives
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Set the output directory to be EXACTLY where the script is located
OUTPUT_DIR = SCRIPT_DIR

# Keep reading the Switch State file from the original Results folder
RESULTS_DIR = PROJECT_ROOT / "Results"
SWITCH_STATE_DIR = RESULTS_DIR / "Switch State"
SWITCH_STATE_FILE = SWITCH_STATE_DIR / "Switch_state.csv"

# Put the logs folder next to the script as well
LOGS_DIR = SCRIPT_DIR / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SWITCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "pipeline.log", encoding='utf-8'), # Added encoding here
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Config:
    PF_PYTHON_PATH = r"C:\Program Files\DIgSILENT\PowerFactory 2023 SP3\Python\3.9"
    PROJECT_NAME = r"\ic84yhos\Venus\VeN2uS_ExampleGrid_v1.4_FRT_KP"
    GRID_NAME = "Grid_110kV.ElmNet"
    DATASET_VERSION = "v1.0"
    DATASET_EXPORT_TYPE = "unified_corridor_feature_matrix"

    REACH_GF = 0.85
    ZONE3_REACH_FACTOR = 1.20
    ZERO_TOLERANCE = 1e-12
    MAX_TRAVERSAL_DEPTH = 50

    ENABLE_LINE_RANDOMIZATION = True
    RANDOMIZED_SCENARIO_COUNT = 10
    RANDOM_SEED_BASE = 1000
    LINE_LENGTH_SCALE_MIN = 0.8
    LINE_LENGTH_SCALE_MAX = 1.2
    INCLUDE_ORIGINAL_BASE_CASE = True

    ENABLE_DG_CAPACITY_RANDOMIZATION = True
    DG_CAPACITY_SCALE_MIN = 0.8
    DG_CAPACITY_SCALE_MAX = 1.2
    DG_CAPACITY_RANDOM_SEED_OFFSET = 100000

    ENABLE_SWITCH_STATE_SCENARIOS = True
    MAX_SWITCH_STATE_CONFIG_COUNT = None
    ENABLE_SWITCH_STATE_OUTSERV_DERIVATION = True


sys.path.append(Config.PF_PYTHON_PATH)
import powerfactory

o_pf = None

# =============================================================================
# Output Schema
# =============================================================================
l_case_feature_columns = [
    "case_id", "relay_id", "relay_node_id", "protected_corridor_id", "subsequent_node_id",
    "line_is_in_service", "corridor_hop_count", "protected_corridor_length_km",
    "protected_corridor_r_ohm", "protected_corridor_x_ohm", "protected_corridor_is_parallel",
    "protected_corridor_parallel_count", "protected_corridor_parallel_id",
    "next_node_busbar_count", "next_node_junction_node_count", "next_node_distributed_generation_count",
    "shortest_downstream_branch_id", "shortest_downstream_branch_remote_node_id",
    "shortest_downstream_branch_hop_count", "shortest_downstream_branch_length_km",
    "shortest_downstream_branch_r_ohm", "shortest_downstream_branch_x_ohm",
    "shortest_downstream_branch_has_parallel", "shortest_downstream_branch_parallel_count",
    "shortest_downstream_branch_parallel_id", "shortest_downstream_branch_junction_node_count",
    "shortest_downstream_branch_junction_node_id", "shortest_downstream_branch_distributed_generation_count",
    "shortest_downstream_branch_distributed_generation_id", "zone2_branch_selection_method",
    "zone2_selected_branch_id", "zone2_reach_calculation_method", "zone2_impedance_basis",
    "parallel_group_count_forward", "parallel_group_id", "zone2_parallel_branch_for_complex_length_km",
    "zone2_parallel_branch_for_complex_r_ohm", "zone2_parallel_branch_for_complex_x_ohm",
    "zone3_branch_selection_method", "zone3_selected_branch_id", "longest_downstream_branch_hop_count",
    "longest_downstream_branch_length_km", "longest_downstream_branch_r_ohm", "longest_downstream_branch_x_ohm",
    "relay_busbar_distributed_generation_count", "relay_busbar_distributed_generation_id",
    "relay_busbar_distributed_generation_capacity_mva", "protected_corridor_distributed_generation_count",
    "protected_corridor_distributed_generation_id", "protected_corridor_distributed_generation_capacity_mva",
    "subsequent_busbar_distributed_generation_count", "subsequent_busbar_distributed_generation_id",
    "subsequent_busbar_distributed_generation_capacity_mva", "downstream_branch_distributed_generation_count",
    "downstream_branch_distributed_generation_id", "downstream_branch_distributed_generation_capacity_mva",
    "remote_busbar_distributed_generation_count", "remote_busbar_distributed_generation_id",
    "remote_busbar_distributed_generation_capacity_mva", "zone1_turbines_candidate_count",
    "zone1_turbines_candidate_id", "zone1_turbines_considered_count", "zone1_turbines_considered_id",
    "zone1_turbines_skipped_count", "zone1_turbines_skipped_id", "zone1_turbines_total_capacity_mva",
    "zone1_total_ikss_contribution_ratio", "zone1_max_single_ikss_contribution_ratio",
    "zone2_turbines_candidate_count", "zone2_turbines_candidate_id", "zone2_turbines_considered_count",
    "zone2_turbines_considered_id", "zone2_turbines_skipped_count", "zone2_turbines_skipped_id",
    "zone2_turbines_total_capacity_mva", "zone2_total_ikss_contribution_ratio",
    "zone2_max_single_ikss_contribution_ratio", "base_zone1_r_reach_ohm", "base_zone1_x_reach_ohm",
    "base_zone2_r_reach_ohm", "base_zone2_x_reach_ohm", "base_zone3_r_reach_ohm", "base_zone3_x_reach_ohm",
    "zone1_infeed_correction_r_ohm", "zone1_infeed_correction_x_ohm", "zone2_infeed_correction_r_ohm",
    "zone2_infeed_correction_x_ohm", "target_zone1_r_reach_ohm", "target_zone1_x_reach_ohm",
    "target_zone2_r_reach_ohm", "target_zone2_x_reach_ohm", "target_zone3_r_reach_ohm", "target_zone3_x_reach_ohm"
]


# =============================================================================
# Basic Utilities
# =============================================================================
def get_boolean_value(b_value): return 1 if bool(b_value) else 0


def get_safe_name(obj):
    try:
        return str(obj.loc_name) if obj else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def get_safe_class_name(obj):
    try:
        return str(obj.GetClassName()) if obj else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def get_safe_full_name(obj):
    try:
        return str(obj.GetFullName()) if obj else get_safe_name(obj)
    except Exception:
        return get_safe_name(obj)


def is_object_in_service(obj):
    try:
        return int(obj.outserv) == 0
    except Exception:
        return True


def is_object_inside_grid(obj, grid):
    try:
        if obj is None or grid is None: return False
        grid_full_name = grid.GetFullName()
        parent = obj
        while parent is not None:
            if parent.GetFullName() == grid_full_name: return True
            parent = parent.GetParent()
        return grid_full_name in obj.GetFullName()
    except Exception:
        return False


def get_pf_attr(obj, attr, default=None, cast_type=None):
    if obj is None: return default
    try:
        val = obj.GetAttribute(attr)
    except Exception:
        try:
            val = getattr(obj, attr)
        except Exception:
            return default
    if val is None: return default
    if cast_type: return cast_type(val)
    return val


def safe_set_attribute(obj, attr, value):
    if obj is None: return False
    try:
        obj.SetAttribute(attr, value)
        return True
    except Exception:
        try:
            setattr(obj, attr, value)
            return True
        except Exception:
            return False


def get_unique_objects(obj_list):
    unique_objs = []
    seen = set()
    for obj in obj_list:
        full_name = get_safe_full_name(obj)
        if full_name not in seen:
            seen.add(full_name)
            unique_objs.append(obj)
    return unique_objs


# =============================================================================
# PF Connection
# =============================================================================
def connect_to_powerfactory():
    logger.info("Connecting to PowerFactory...")
    try:
        app = powerfactory.GetApplicationExt()
    except Exception:
        app = powerfactory.GetApplication()
    if app is None: raise RuntimeError("PowerFactory connection failed.")
    app.Show()
    logger.info("PowerFactory connected.")
    return app


def activate_powerfactory_project(app, project_name):
    logger.info(f"Activating project: {project_name}")
    if app.ActivateProject(project_name) != 0: raise RuntimeError(f"Could not activate project: {project_name}")
    proj = app.GetActiveProject()
    if proj is None: raise RuntimeError("No active PowerFactory project found.")
    return proj


def get_target_grid(proj, grid_name):
    logger.info(f"Selecting target grid: {grid_name}")
    grids = proj.GetContents(grid_name, 1)
    if not grids: raise RuntimeError(f"Grid not found: {grid_name}")
    return grids[0]


# =============================================================================
# Topology Core
# =============================================================================
def get_terminal_from_cubicle(cubicle): return get_pf_attr(cubicle, "cterm")


def get_line_terminals(line):
    b1 = get_pf_attr(line, "bus1")
    b2 = get_pf_attr(line, "bus2")
    return get_terminal_from_cubicle(b1), get_terminal_from_cubicle(b2), b1, b2


def get_opposite_terminal(line, terminal):
    t1, t2, _, _ = get_line_terminals(line)
    if t1 == terminal: return t2
    if t2 == terminal: return t1
    return None


def get_line_impedance(line): return get_pf_attr(line, "R1", 0.0, float), get_pf_attr(line, "X1", 0.0, float)


def get_line_length_value(line): return get_pf_attr(line, "dline", 0.0, float)


def get_cubicle_switch_closed_state(cubicle):
    if cubicle is None: return 0
    try:
        sws = cubicle.GetChildren(1, "*.StaSwitch") or []
        if sws: return 1 if get_pf_attr(sws[0], "on_off", 1, int) == 1 else 0
    except Exception:
        pass
    return 1


def get_line_is_available_for_topology(line):
    if line is None or not is_object_in_service(line): return False
    _, _, c1, c2 = get_line_terminals(line)
    return get_cubicle_switch_closed_state(c1) == 1 and get_cubicle_switch_closed_state(c2) == 1


def get_terminal_is_busbar(terminal): return get_pf_attr(terminal, "iUsage", 0, int) == 0


def get_terminal_is_junction_node(terminal): return get_pf_attr(terminal, "iUsage", 0, int) == 1


def get_terminal_connected_elements(terminal):
    try:
        return list(terminal.GetConnectedElements())
    except Exception:
        return []


def get_terminal_connected_lines(terminal):
    return [e for e in get_terminal_connected_elements(terminal) if
            get_safe_class_name(e) == "ElmLne" and get_line_is_available_for_topology(e)]


def get_terminal_connected_loads(terminal):
    return [e for e in get_terminal_connected_elements(terminal) if
            get_safe_class_name(e) in ["ElmLod", "ElmLodlv"] and is_object_in_service(e)]


def get_terminal_connected_distributed_generators(terminal):
    return [e for e in get_terminal_connected_elements(terminal) if
            get_safe_class_name(e) in ["ElmGenstat", "ElmPvsys"] and is_object_in_service(e)]


def get_terminal_has_load(terminal): return len(get_terminal_connected_loads(terminal)) > 0


def get_terminal_load_p_mw(terminal):
    return sum([get_pf_attr(l, "plini", get_pf_attr(l, "pgini", 0.0, float), float) for l in
                get_terminal_connected_loads(terminal)])


def get_object_is_relay(obj):
    sn = get_safe_class_name(obj).lower() + get_safe_name(obj).lower()
    return any(m in sn for m in ["relay", "rel", "reldis", "distance", "reloc", "elmrelay"])


def get_terminal_connected_relays(terminal):
    relays = [e for e in get_terminal_connected_elements(terminal) if get_object_is_relay(e)]
    for line in get_terminal_connected_lines(terminal):
        t1, t2, c1, c2 = get_line_terminals(line)
        cub = c1 if t1 == terminal else c2
        if cub:
            try:
                contents = cub.GetContents("*", 1)
            except Exception:
                contents = []
            relays.extend([e for e in contents if get_object_is_relay(e)])
    return get_unique_objects(relays)


def get_relay_id_from_terminal_and_line(terminal, line):
    # 1. Find the specific cubicle for THIS line at THIS terminal
    cubicle = get_cubicle_for_line_at_terminal(line, terminal)

    if cubicle:
        try:
            # 2. Get the contents of only this specific cubicle
            contents = cubicle.GetContents("*", 1)
        except Exception:
            contents = []

        # 3. Look for a relay inside this cubicle
        relays = [e for e in contents if get_object_is_relay(e)]

        if relays:
            return get_safe_name(relays[0])

    # 4. Fallback if no specific relay is found
    return f"relay_node_{get_safe_name(terminal)}_line_{get_safe_name(line)}"


def get_cubicle_for_line_at_terminal(line, terminal):
    t1, t2, c1, c2 = get_line_terminals(line)
    if t1 == terminal: return c1
    if t2 == terminal: return c2
    return None


def line_connects_terminal_pair(line, term_a_name, term_b_name):
    t1, t2, _, _ = get_line_terminals(line)
    if not t1 or not t2: return False
    return {get_safe_full_name(t1), get_safe_full_name(t2)} == {term_a_name, term_b_name}


def get_parallel_lines_for_line(line):
    if not line: return []
    t1, t2, _, _ = get_line_terminals(line)
    if not t1 or not t2: return []
    t1n, t2n = get_safe_full_name(t1), get_safe_full_name(t2)
    return get_unique_objects([cand for cand in get_terminal_connected_lines(t1) if
                               cand != line and line_connects_terminal_pair(cand, t1n, t2n)])


def get_parallel_lines_for_line_list(lines):
    parallels = []
    for l in lines: parallels.extend(get_parallel_lines_for_line(l))
    return get_unique_objects(parallels)


def get_branch_line_names(lines): return " | ".join([get_safe_name(l) for l in lines if l])


# =============================================================================
# DG Topology & Summaries
# =============================================================================
def get_unique_distributed_generators_from_terminals(terminals):
    dgs = []
    for t in terminals: dgs.extend(get_terminal_connected_distributed_generators(t))
    return get_unique_objects(dgs)


def get_distributed_generator_capacity_mva(dg):
    for a in ["sgn", "Sn", "snom", "Srated"]:
        val = get_pf_attr(dg, a, None, float)
        if val is not None: return val
    return 0.0


def get_connected_terminal_for_distributed_generator(o_dg):
    """Returns the terminal connected to a distributed generator."""
    cubicle = get_pf_attr(o_dg, "bus1")
    return get_terminal_from_cubicle(cubicle)


def get_dg_summary_from_terminals(terminals):
    dgs = get_unique_distributed_generators_from_terminals(terminals)
    return {
        "has_dg": get_boolean_value(len(dgs) > 0),
        "count": len(dgs),
        "capacity_mva": round(sum([get_distributed_generator_capacity_mva(dg) for dg in dgs]), 3),
        "names": " -> ".join([get_safe_name(dg) for dg in dgs]),
        "objects": dgs,
    }


def get_ordered_terminals_along_protected_corridor(corridor):
    terms = []
    cur = corridor.get("relay_busbar")
    if not cur: return terms
    terms.append(cur)
    for line in corridor.get("line_sections", []):
        nxt = get_opposite_terminal(line, cur)
        if not nxt: break
        terms.append(nxt)
        cur = nxt
    return terms


def summarize_dg_by_corridor_location(corridor, shortest_branch):
    rb = corridor.get("relay_busbar")
    sb = corridor.get("subsequent_busbar")
    cterms = get_ordered_terminals_along_protected_corridor(corridor)
    bterms = shortest_branch.get("Branch Terminals", [])

    return {
        "relay_busbar": get_dg_summary_from_terminals([rb] if rb else []),
        "subsequent_busbar": get_dg_summary_from_terminals([sb] if sb else []),
        "protected_corridor": get_dg_summary_from_terminals([t for t in cterms if t not in (rb, sb)]),
        "downstream_branch": get_dg_summary_from_terminals(bterms[1:-1] if len(bterms) > 2 else []),
        "remote_busbar": get_dg_summary_from_terminals([bterms[-1]] if len(bterms) > 1 else [])
    }


def split_protected_corridor_turbines_by_zone_reach(corridor, turbines, reach_fraction):
    rb = corridor.get("relay_busbar")
    lines = corridor.get("line_sections", [])
    segments = build_ordered_path_segments(rb, lines)
    if not segments: return [], []

    z1_dist = float(reach_fraction or 0.0) * segments[-1].get("end_distance_km", 0.0)
    z1_t, z2_t = [], []

    for dg in turbines:
        dgt = get_pf_attr(get_pf_attr(dg, "bus1"), "cterm")
        dist = get_terminal_distance_on_ordered_path(dgt, segments)
        if dist is None: continue
        if dist <= z1_dist + 1e-9:
            z1_t.append(dg)
        else:
            z2_t.append(dg)
    return get_unique_objects(z1_t), get_unique_objects(z2_t)


# =============================================================================
# Network Extraction
# =============================================================================
def extract_network_once(grid):
    logger.info("Extracting line data...")
    lines = [l for l in o_pf.GetCalcRelevantObjects("*.ElmLne") or [] if get_line_is_available_for_topology(l)]
    line_data, skipped = [], []
    for line in lines:
        try:
            cf, ct = get_pf_attr(line, "bus1"), get_pf_attr(line, "bus2")
            if not cf or not ct: continue
            tf, tt = get_terminal_from_cubicle(cf), get_terminal_from_cubicle(ct)
            if not tf or not tt or not is_object_inside_grid(tf, grid) or not is_object_inside_grid(tt, grid): continue

            line_data.append({
                "_pf_line_object": line,
                "_pf_terminal_from": tf,
                "_pf_terminal_to": tt,
                "_pf_cubicle_from": cf,
                "_pf_cubicle_to": ct
            })
        except Exception as e:
            skipped.append((get_safe_name(line), str(e)))

    logger.info(f"Collected {len(line_data)} valid lines.")
    return {"line_data": line_data, "skipped_lines": skipped}


def calculate_line_path_impedance(lines):
    r, x, l = 0.0, 0.0, 0.0
    for line in lines:
        lr, lx = get_line_impedance(line)
        r += lr;
        x += lx;
        l += get_line_length_value(line)
    return {"total_r_ohm": round(r, 3), "total_x_ohm": round(x, 3), "total_length_km": round(l, 3),
            "hop_count": len(lines)}


def get_next_line_from_junction(junction, prev_line, visited):
    for cand in get_terminal_connected_lines(junction):
        if cand != prev_line and cand not in visited: return cand
    return None


def trace_protected_corridor_from_relay_busbar(rb, first_line):
    if not get_terminal_is_busbar(rb): return None
    lines, jnodes, visited = [], [], []
    cur_term, cur_line = rb, first_line
    while cur_line and cur_line not in visited:
        visited.append(cur_line)
        lines.append(cur_line)
        nxt_term = get_opposite_terminal(cur_line, cur_term)
        if not nxt_term: break
        if get_terminal_is_busbar(nxt_term):
            imp = calculate_line_path_impedance(lines)
            return {
                "relay_busbar": rb, "subsequent_busbar": nxt_term,
                "line_sections": lines, "junction_nodes": jnodes,
                "first_line_section": lines[0], "last_line_section": lines[-1],
                "protected_corridor_id": get_branch_line_names(lines),
                **imp
            }
        if get_terminal_is_junction_node(nxt_term):
            jnodes.append(nxt_term)
            cur_line = get_next_line_from_junction(nxt_term, cur_line, visited)
            cur_term = nxt_term
            continue
        break
    return None


def build_all_directional_protected_corridors(line_data):
    corridors = []
    seen = set()
    for d in line_data:
        line, tf, tt = d["_pf_line_object"], d["_pf_terminal_from"], d["_pf_terminal_to"]
        for rb in [tf, tt]:
            if not get_terminal_is_busbar(rb): continue
            corr = trace_protected_corridor_from_relay_busbar(rb, line)
            if not corr: continue
            key = (get_safe_full_name(corr["relay_busbar"]), get_safe_full_name(corr["subsequent_busbar"]),
                   corr["protected_corridor_id"])
            if key not in seen:
                corr["relay_cubicle"] = get_cubicle_for_line_at_terminal(corr["first_line_section"],
                                                                         corr["relay_busbar"])
                seen.add(key)
                corridors.append(corr)
    logger.info(f"Protected corridors detected: {len(corridors)}")
    return corridors


def detect_protected_corridors_once(net):
    return build_all_directional_protected_corridors(net.get("line_data", []))


def trace_branch_from_busbar_to_next_busbar(sb, first_line, excluded):
    lines, terms, visited = [], [sb], excluded[:]
    cur_term, cur_line = sb, first_line
    while cur_line and cur_line not in visited:
        visited.append(cur_line);
        lines.append(cur_line)
        nxt_term = get_opposite_terminal(cur_line, cur_term)
        if not nxt_term: break
        terms.append(nxt_term)
        if get_terminal_is_busbar(nxt_term): return build_branch_summary(lines, terms)
        if get_terminal_is_junction_node(nxt_term):
            cur_line = get_next_line_from_junction(nxt_term, cur_line, visited)
            cur_term = nxt_term
            continue
        break
    return None


def build_empty_branch_summary(sb):
    return {
        "Branch Lines": [], "Branch Terminals": [sb] if sb else [], "Branch ID": "", "Remote Node ID": "",
        "Hop Count": 0, "Branch Length": 0.0, "Branch Resistance": 0.0, "Branch Reactance": 0.0,
        "Has Parallel": False, "Parallel Count": 0, "Parallel Lines": "", "Junction Node Count": 0,
        "Distributed Generation Count": 0
    }


def get_branch_far_end_busbar(branch):
    terms = branch.get("Branch Terminals", [])
    return terms[-1] if terms else None


def build_branch_summary(lines, terms):
    imp = calculate_line_path_impedance(lines)
    plines = get_parallel_lines_for_line_list(lines)
    jnodes = [t for t in terms if get_terminal_is_junction_node(t)]
    dgs = get_unique_distributed_generators_from_terminals(terms)
    return {
        "Branch Lines": lines, "Branch Terminals": terms,
        "Branch ID": get_branch_line_names(lines),
        "Remote Node ID": get_safe_name(terms[-1]) if terms else "",
        "Hop Count": len(lines), "Branch Length": imp.get("total_length_km", 0.0),
        "Branch Resistance": imp.get("total_r_ohm", 0.0), "Branch Reactance": imp.get("total_x_ohm", 0.0),
        "Has Parallel": len(plines) > 0, "Parallel Count": len(plines),
        "Parallel Lines": get_branch_line_names(plines),
        "Junction Node Count": len(jnodes), "Distributed Generation Count": len(dgs)
    }


def filter_valid_downstream_branches_excluding_relay_return(corr, all_corridors=None):
    rb = corr.get("relay_busbar")
    sb = corr.get("subsequent_busbar")
    exc = corr.get("line_sections", [])[:]
    if not rb or not sb: return []

    if all_corridors:
        for pcorr in find_parallel_protected_corridors(corr, all_corridors):
            for l in pcorr.get("line_sections", []):
                if l not in exc: exc.append(l)

    valid = []
    for cand in get_terminal_connected_lines(sb):
        if cand in exc or line_connects_terminal_pair(cand, get_safe_full_name(rb), get_safe_full_name(sb)): continue
        branch = trace_branch_from_busbar_to_next_busbar(sb, cand, exc[:])
        if not branch: continue
        fb = get_branch_far_end_busbar(branch)
        if not fb or get_safe_full_name(fb) in [get_safe_full_name(rb), get_safe_full_name(sb)]: continue
        valid.append(branch)
    return valid


def summarize_next_node_downstream_context(branches):
    rb, jn, dg = [], [], []
    for b in branches:
        terms = b.get("Branch Terminals", [])
        if terms and get_terminal_is_busbar(terms[-1]): rb.append(terms[-1])
        jn.extend([t for t in terms if get_terminal_is_junction_node(t)])
        dg.extend(get_unique_distributed_generators_from_terminals(terms))
    return {
        "busbar_count": len(get_unique_objects(rb)),
        "junction_node_count": len(get_unique_objects(jn)),
        "distributed_generation_count": len(get_unique_objects(dg))
    }


def protected_corridors_have_same_busbar_pair(ca, cb):
    if not all([ca.get("relay_busbar"), ca.get("subsequent_busbar"), cb.get("relay_busbar"),
                cb.get("subsequent_busbar")]): return False
    pa = {get_safe_full_name(ca.get("relay_busbar")), get_safe_full_name(ca.get("subsequent_busbar"))}
    pb = {get_safe_full_name(cb.get("relay_busbar")), get_safe_full_name(cb.get("subsequent_busbar"))}
    return pa == pb


def get_protected_corridor_line_full_name_set(corr):
    return frozenset([get_safe_full_name(l) for l in corr.get("line_sections", [])])


def find_parallel_protected_corridors(tcorr, all_corrs):
    tset = get_protected_corridor_line_full_name_set(tcorr)
    return [c for c in all_corrs if c != tcorr and protected_corridors_have_same_busbar_pair(tcorr,
                                                                                             c) and get_protected_corridor_line_full_name_set(
        c) != tset]


def summarize_parallel_protected_corridors(tcorr, all_corrs):
    pcorrs = find_parallel_protected_corridors(tcorr, all_corrs)
    fc = pcorrs[0] if pcorrs else {}
    return {
        "is_parallel": len(pcorrs) > 0, "parallel_count": len(pcorrs) + 1 if pcorrs else 1,
        "parallel_alternative_lines": get_branch_line_names(fc.get("line_sections", [])),
        "parallel_r_ohm": fc.get("total_r_ohm", 0.0), "parallel_x_ohm": fc.get("total_x_ohm", 0.0),
        "parallel_length_km": fc.get("total_length_km", 0.0)
    }


def get_physical_corridor_line_key(corr):
    return " || ".join(sorted([get_safe_full_name(l) for l in corr.get("line_sections", []) if l]))


def get_parallel_protected_corridor_line_names(corr, all_corrs):
    tpair = frozenset([get_safe_full_name(corr.get("relay_busbar")), get_safe_full_name(corr.get("subsequent_busbar"))])
    tkey = get_physical_corridor_line_key(corr)
    names = {}
    for c in all_corrs:
        cpair = frozenset([get_safe_full_name(c.get("relay_busbar")), get_safe_full_name(c.get("subsequent_busbar"))])
        ckey = get_physical_corridor_line_key(c)
        if cpair == tpair and ckey and ckey != tkey:
            names[ckey] = get_branch_line_names(c.get("line_sections", []))
    return " | ".join(sorted(names.values()))


# =============================================================================
# Zone Selection & Math
# =============================================================================
def select_zone2_downstream_branch_group(corr, l_all_corridors=None):
    sb = corr.get("subsequent_busbar")
    valid = filter_valid_downstream_branches_excluding_relay_return(corr, l_all_corridors)
    if not valid:
        return {
            "selected_branch": build_empty_branch_summary(sb), "selected_parallel_branches": [],
            "selected_remote_busbar_name": "", "zone2_branch_selection_method": "no_valid_downstream_branch",
            "zone2_reach_calculation_method": "simple_no_downstream_branch", "zone2_selected_branch_has_parallel": 0,
            "zone2_selected_branch_parallel_count": 0
        }

    grouped = defaultdict(list)
    for b in valid: grouped[get_safe_full_name(get_branch_far_end_busbar(b))].append(b)

    summaries = []
    for k, branches in grouped.items():
        sorted_b = sorted(branches, key=lambda d: (
        float(d.get("Branch Reactance", 0.0)), float(d.get("Branch Length", 0.0)),
        float(d.get("Branch Resistance", 0.0))))
        summaries.append({
            "selected_branch": sorted_b[0], "parallel_branches": sorted_b,
            "parallel_count": len(sorted_b), "remote_busbar": get_branch_far_end_busbar(sorted_b[0])
        })

    summaries = sorted(summaries, key=lambda d: (
    float(d["selected_branch"].get("Branch Reactance", 0.0)), float(d["selected_branch"].get("Branch Length", 0.0))))
    sg = summaries[0]
    has_p = sg["parallel_count"] > 1
    return {
        "selected_branch": sg["selected_branch"], "selected_parallel_branches": sg["parallel_branches"],
        "selected_remote_busbar_name": get_safe_name(sg["remote_busbar"]),
        "zone2_branch_selection_method": "shortest_valid_remote_busbar_group_with_parallel" if has_p else "shortest_valid_downstream_branch",
        "zone2_reach_calculation_method": "complex_parallel_zone2_reach" if has_p else "simple_zone2_reach",
        "zone2_selected_branch_has_parallel": get_boolean_value(has_p),
        "zone2_selected_branch_parallel_count": sg["parallel_count"]
    }


def select_parallel_branch_for_complex_zone2(sel_branch, par_branches):
    cands = [b for b in par_branches if
             b != sel_branch and get_branch_line_names(b.get("Branch Lines", [])) != get_branch_line_names(
                 sel_branch.get("Branch Lines", []))]
    if not cands: return None
    return sorted(cands, key=lambda d: (float(d.get("Branch Reactance", 0.0)), float(d.get("Branch Length", 0.0))))[0]


def select_zone3_longest_valid_downstream_branch(corr, l_all_corridors=None):
    valid = filter_valid_downstream_branches_excluding_relay_return(corr, l_all_corridors)
    if not valid: return build_empty_branch_summary(corr.get("subsequent_busbar"))
    return sorted(valid, key=lambda d: (float(d.get("Branch Length", 0.0)), float(d.get("Branch Reactance", 0.0))),
                  reverse=True)[0]


def count_forward_parallel_branch_groups_for_corridor(corr, l_all_corridors=None):
    valid = filter_valid_downstream_branches_excluding_relay_return(corr, l_all_corridors)
    grouped = defaultdict(list)
    for b in valid: grouped[get_safe_full_name(get_branch_far_end_busbar(b))].append(b)
    pgc, mpc = 0, 0
    for _, branches in grouped.items():
        if len(branches) > 1:
            pgc += 1
            mpc = max(mpc, len(branches))
    return pgc, mpc


def calculate_simple_zone2_reach(r_conn, x_conn, r_branch, x_branch):
    gf = Config.REACH_GF
    return round(gf * (r_conn + gf * r_branch), 3), round(gf * (x_conn + gf * x_branch), 3)


def calculate_complex_zone2_reach(r_conn, x_conn, r_short, x_short, r_parallel, x_parallel):
    z1, z2, z3 = complex(r_conn, x_conn), complex(r_short, x_short), complex(r_parallel, x_parallel)
    if abs(z2 + z3) <= Config.ZERO_TOLERANCE: return calculate_simple_zone2_reach(r_conn, x_conn, r_short, x_short)
    gf = Config.REACH_GF
    zc = gf * (z1 + ((z3 + (1.0 - gf) * z2) * gf * z2) / (z2 + z3))
    return round(zc.real, 3), round(zc.imag, 3)


def calculate_distance_zone_reaches_for_corridor(corr, z2_branch, p_summary, all_corrs=None):
    r_corr = float(corr.get("total_r_ohm", 0.0))
    x_corr = float(corr.get("total_x_ohm", 0.0))
    l_corr = float(corr.get("total_length_km", 0.0))

    z1_r = round(Config.REACH_GF * r_corr, 3)
    z1_x = round(Config.REACH_GF * x_corr, 3)

    z2_sel = select_zone2_downstream_branch_group(corr, all_corrs)
    z2_b = z2_sel.get("selected_branch", build_empty_branch_summary(corr.get("subsequent_busbar")))
    z2_pb = z2_sel.get("selected_parallel_branches", [])

    r_z2b = float(z2_b.get("Branch Resistance", 0.0))
    x_z2b = float(z2_b.get("Branch Reactance", 0.0))
    l_z2b = float(z2_b.get("Branch Length", 0.0))

    int_z2_r, int_z2_x = calculate_simple_zone2_reach(r_corr, x_corr, r_z2b, x_z2b)

    z2_pr, z2_px, z2_pl, z2_pid = 0.0, 0.0, 0.0, ""
    has_p = bool(z2_sel.get("zone2_selected_branch_has_parallel", 0))

    if has_p and len(z2_pb) > 1:
        z2_cplex = select_parallel_branch_for_complex_zone2(z2_b, z2_pb)
        if z2_cplex:
            z2_pr = float(z2_cplex.get("Branch Resistance", 0.0))
            z2_px = float(z2_cplex.get("Branch Reactance", 0.0))
            z2_pl = float(z2_cplex.get("Branch Length", 0.0))
            z2_pid = get_branch_line_names(z2_cplex.get("Branch Lines", []))
            z2_r, z2_x = calculate_complex_zone2_reach(r_corr, x_corr, r_z2b, x_z2b, z2_pr, z2_px)
            zb = "complex_parallel_shortest_branch_and_parallel_branch_impedance"
        else:
            z2_r, z2_x = calculate_simple_zone2_reach(r_corr, x_corr, r_z2b, x_z2b)
            zb = "simple_selected_downstream_branch_impedance_no_parallel_alternative_found"
    else:
        z2_r, z2_x = calculate_simple_zone2_reach(r_corr, x_corr, r_z2b, x_z2b)
        zb = "simple_selected_downstream_branch_impedance"

    z3_b = select_zone3_longest_valid_downstream_branch(corr, all_corrs)
    r_z3b = float(z3_b.get("Branch Resistance", 0.0))
    x_z3b = float(z3_b.get("Branch Reactance", 0.0))

    z3_r = round(r_corr + (Config.ZONE3_REACH_FACTOR * r_z3b), 3)
    z3_x = round(x_corr + (Config.ZONE3_REACH_FACTOR * x_z3b), 3)

    return {
        "protected_corridor_r_ohm": round(r_corr, 3), "protected_corridor_x_ohm": round(x_corr, 3),
        "protected_corridor_length_km": round(l_corr, 3), "zone1_r_reach_ohm": z1_r, "zone1_x_reach_ohm": z1_x,
        "zone2_r_reach_ohm": z2_r, "zone2_x_reach_ohm": z2_x, "intended_zone2_r_reach_ohm": int_z2_r,
        "intended_zone2_x_reach_ohm": int_z2_x,
        "zone2_reach_calculation_method": z2_sel.get("zone2_reach_calculation_method", ""),
        "zone2_branch_selection_method": z2_sel.get("zone2_branch_selection_method", ""),
        "zone2_selected_remote_busbar": z2_sel.get("selected_remote_busbar_name", ""),
        "zone2_downstream_branch_id": get_branch_line_names(z2_b.get("Branch Lines", [])),
        "zone2_downstream_branch_length_km": round(l_z2b, 3), "zone2_downstream_branch_r_ohm": round(r_z2b, 3),
        "zone2_downstream_branch_x_ohm": round(x_z2b, 3),
        "zone2_selected_branch_has_parallel": get_boolean_value(has_p),
        "zone2_selected_branch_parallel_count": int(z2_sel.get("zone2_selected_branch_parallel_count", 0)),
        "zone2_impedance_basis": zb, "zone2_branch_for_reach_r_ohm": round(r_z2b, 3),
        "zone2_branch_for_reach_x_ohm": round(x_z2b, 3),
        "zone2_parallel_branch_for_complex_id": z2_pid, "zone2_parallel_branch_for_complex_length_km": round(z2_pl, 3),
        "zone2_parallel_branch_for_complex_r_ohm": round(z2_pr, 3),
        "zone2_parallel_branch_for_complex_x_ohm": round(z2_px, 3),
        "zone3_r_reach_ohm": z3_r, "zone3_x_reach_ohm": z3_x,
        "zone3_branch_selection_method": "longest_valid_downstream_branch_excluding_relay_return",
        "zone3_downstream_branch_id": get_branch_line_names(z3_b.get("Branch Lines", [])),
        "zone3_downstream_branch_length_km": round(float(z3_b.get("Branch Length", 0.0)), 3),
        "zone3_downstream_branch_r_ohm": round(r_z3b, 3), "zone3_downstream_branch_x_ohm": round(x_z3b, 3),
        "protected_corridor_is_parallel": get_boolean_value(bool(p_summary.get("is_parallel", False))),
        "protected_corridor_parallel_count": int(p_summary.get("parallel_count", 1)),
        "protected_corridor_parallel_alternative_lines": p_summary.get("parallel_alternative_lines", "")
    }


# =============================================================================
# Short Circuit & Infeed Calculations
# =============================================================================
def get_short_circuit_command(proj):
    try:
        shc = o_pf.GetFromStudyCase("ComShc")
        if shc: return shc
    except Exception:
        pass
    try:
        shcs = proj.GetContents("Short-Circuit Calculation*", 1)
        if shcs: return shcs[0]
    except Exception:
        pass
    return None


def configure_short_circuit_command(shc):
    if not shc: return
    for attr, val in [("iopt_allbus", 0), ("iopt_mode", 3), ("iopt_mde", 3), ("ip_flt", 0), ("iopt_opt", "pro"),
                      ("iopt_cur", 0)]:
        safe_set_attribute(shc, attr, val)


def execute_short_circuit_at_line_fault(proj, line, pct, dist_km=None):
    if not line: return False
    shc = get_short_circuit_command(proj)
    if not shc: return False
    configure_short_circuit_command(shc)
    safe_set_attribute(shc, "shcobj", line)
    for a in ["ppro", "relpos", "xloc", "fltloc", "loc_fault"]: safe_set_attribute(shc, a, float(pct or 0.0))
    if dist_km is not None: safe_set_attribute(shc, "faultloc", float(dist_km or 0.0))
    try:
        return shc.Execute() == 0
    except Exception:
        return False


def get_ikss_value(obj):
    for a in ["m:Ikss:bus1", "m:Ikss:bus2", "m:Ikss"]:
        v = get_pf_attr(obj, a, None, float)
        if v is not None: return v
    return 0.0


def _get_relay_side_and_cubicle(line, start_term):
    try:
        cb1, cb2 = get_pf_attr(line, "bus1"), get_pf_attr(line, "bus2")
        tb1, tb2 = get_terminal_from_cubicle(cb1), get_terminal_from_cubicle(cb2)
        if tb1 == start_term: return cb1, get_safe_name(cb1), "m:Ikss:bus1", "bus1"
        if tb2 == start_term: return cb2, get_safe_name(cb2), "m:Ikss:bus2", "bus2"
    except Exception:
        pass
    return None, "UNKNOWN", "m:Ikss:bus1", "unknown"


def _read_relay_reference_ikss(ref_cub, ref_line, ikss_attr):
    v = get_pf_attr(ref_line, ikss_attr, None, float)
    if v and v > 0.0: return v, f"line.{ikss_attr}"
    v = get_pf_attr(ref_line, "m:Ikss", None, float)
    if v and v > 0.0: return v, "line.m:Ikss"
    if ref_cub:
        for a in ["m:Ikss", "m:Ikss:bus1", "m:Ikss:bus2"]:
            v = get_pf_attr(ref_cub, a, None, float)
            if v and v > 0.0: return v, f"cubicle.{a}"
    return 0.0, "NOT_FOUND"


def build_ordered_path_segments(start_term, lines):
    segments = []
    if not start_term or not lines: return segments
    cur, dist = start_term, 0.0
    for l in lines:
        nxt = get_opposite_terminal(l, cur)
        if not nxt: break
        lkm = get_line_length_value(l)
        r, x = get_line_impedance(l)
        segments.append({
            "line": l, "start_terminal": cur, "end_terminal": nxt,
            "start_distance_km": dist, "end_distance_km": dist + lkm,
            "length_km": lkm, "r_ohm": r, "x_ohm": x
        })
        dist += lkm;
        cur = nxt
    return segments


def get_terminal_distance_on_ordered_path(term, segments):
    if not term: return None
    for s in segments:
        if term == s["start_terminal"]: return s["start_distance_km"]
        if term == s["end_terminal"]: return s["end_distance_km"]
    return None


def calculate_path_impedance_between_distances(segments, d_from, d_to):
    if d_from is None or d_to is None: return 0.0, 0.0
    s_start, s_end = min(d_from, d_to), max(d_from, d_to)
    r, x = 0.0, 0.0
    for s in segments:
        os, oe = max(s_start, s["start_distance_km"]), min(s_end, s["end_distance_km"])
        if oe <= os or s["length_km"] <= 0.0: continue
        frac = (oe - os) / s["length_km"]
        r += s["r_ohm"] * frac;
        x += s["x_ohm"] * frac
    return round(r, 3), round(x, 3)


def select_fault_location_by_reach_impedance(segments, tgt_r, tgt_x):
    if not segments: return None, 0.0, 0.0, 0.0
    tgt_x, cx = max(0.0, float(tgt_x or 0.0)), 0.0
    for s in segments:
        sx, skm, sst = float(s.get("x_ohm", 0.0)), float(s.get("length_km", 0.0)), float(
            s.get("start_distance_km", 0.0))
        if sx > 0.0 and (cx + sx >= tgt_x):
            frac = max(0.0, min(1.0, (tgt_x - cx) / sx))
            lkm = frac * skm
            return s["line"], 100.0 * frac, lkm, sst + lkm
        cx += sx
    ls = segments[-1]
    return ls["line"], 100.0, ls.get("length_km", 0.0), ls.get("end_distance_km", 0.0)


def select_shc_fault_location_for_dg_context(dg_term, segments):
    if not dg_term or not segments: return None
    if get_terminal_is_busbar(dg_term):
        ds_segs, coll = [], False
        for s in segments:
            if not coll:
                if s.get("start_terminal") != dg_term: continue
                coll = True
            ds_segs.append(s)
            end_t = s.get("end_terminal")
            if end_t != dg_term and get_terminal_is_busbar(end_t): break
        if not ds_segs: return None
        tot_l = sum([float(s.get("length_km", 0.0)) for s in ds_segs])
        if tot_l <= 0.0: return None
        mid = 0.5 * tot_l
        acc = 0.0
        for s in ds_segs:
            sl = float(s.get("length_km", 0.0))
            if acc + sl >= mid:
                loc = mid - acc
                pct = 100.0 * loc / sl if sl > 0 else 0.0
                return {"fault_line": s.get("line"), "fault_percent": pct, "local_fault_distance_km": loc,
                        "fault_distance_km": float(s.get("start_distance_km", 0.0)) + loc, "fault_rule": "busbar_dg"}
            acc += sl
        ls = ds_segs[-1]
        return {"fault_line": ls.get("line"), "fault_percent": 100.0,
                "local_fault_distance_km": float(ls.get("length_km", 0.0)),
                "fault_distance_km": float(ls.get("end_distance_km", 0.0)), "fault_rule": "busbar_dg"}

    if get_terminal_is_junction_node(dg_term):
        s_idx = next((i for i, s in enumerate(segments) if
                      s.get("start_terminal") == dg_term or s.get("end_terminal") == dg_term), None)
        if s_idx is None: return None
        if segments[s_idx].get("end_terminal") == dg_term: s_idx += 1
        lsec = None
        for s in segments[s_idx:]:
            lsec = s
            if get_terminal_is_busbar(s.get("end_terminal")): break
        if not lsec: return None
        loc = 0.5 * float(lsec.get("length_km", 0.0))
        return {"fault_line": lsec.get("line"), "fault_percent": 50.0, "local_fault_distance_km": loc,
                "fault_distance_km": float(lsec.get("start_distance_km", 0.0)) + loc, "fault_rule": "junction_dg"}
    return None


def get_all_grid_distributed_generators(grid):
    dgs = []
    for f in ["*.ElmGenstat", "*.ElmPvsys"]:
        try:
            objs = o_pf.GetCalcRelevantObjects(f) or []
        except Exception:
            objs = []
        for obj in objs:
            if not is_object_in_service(obj): continue
            cub = get_pf_attr(obj, "bus1")
            if get_cubicle_switch_closed_state(cub) == 0: continue
            term = get_terminal_from_cubicle(cub)
            if is_object_inside_grid(term, grid): dgs.append(obj)
    return get_unique_objects(dgs)


def get_original_outserv_states(objs):
    return {o: get_pf_attr(o, "outserv", 0, int) for o in objs}


def restore_original_outserv_states(states):
    for o, v in states.items(): safe_set_attribute(o, "outserv", v)


def activate_only_one_distributed_generator(all_dg, active_dg):
    for dg in all_dg: safe_set_attribute(dg, "outserv", 0 if dg == active_dg else 1)


def calculate_zone_infeed_summary_for_turbines(
        o_project, o_grid, o_start_terminal, o_reference_cubicle, o_reference_line,
        s_relay_ikss_line_attr, l_turbines, l_fault_lines, f_fault_reach_r_ohm,
        f_fault_reach_x_ohm, f_relay_reference_ikss_ka, s_zone_name):
    d_summary = {
        "turbines_candidate_count": 0, "turbines_candidate_id": "",
        "turbines_considered_count": 0, "turbines_considered_id": "",
        "turbines_skipped_count": 0, "turbines_skipped_id": "",
        "turbines_total_capacity_mva": 0.0, "total_ikss_contribution_ratio": 0.0,
        "max_single_ikss_contribution_ratio": 0.0, "infeed_correction_r_ohm": 0.0,
        "infeed_correction_x_ohm": 0.0, "relay_reference_ikss_ka": f_relay_reference_ikss_ka,
    }

    l_turbines = get_unique_objects(l_turbines)
    d_summary["turbines_candidate_count"] = len(l_turbines)
    d_summary["turbines_candidate_id"] = get_object_id_string(l_turbines)
    if not l_turbines: return d_summary

    l_path_segments = build_ordered_path_segments(o_start_terminal, l_fault_lines)
    if not l_path_segments: return d_summary

    _, _, _, f_zone_boundary_distance_km = select_fault_location_by_reach_impedance(
        l_path_segments, f_fault_reach_r_ohm, f_fault_reach_x_ohm
    )

    l_valid_turbine_items = []
    l_skipped_turbines = []

    for o_dg in l_turbines:
        o_dg_terminal = get_connected_terminal_for_distributed_generator(o_dg)
        f_dg_distance_km = get_terminal_distance_on_ordered_path(o_dg_terminal, l_path_segments)

        if f_dg_distance_km is None or f_dg_distance_km > f_zone_boundary_distance_km + 1e-9:
            l_skipped_turbines.append(o_dg)
            continue

        d_fault_context = select_shc_fault_location_for_dg_context(o_dg_terminal, l_path_segments)
        if d_fault_context is None:
            l_skipped_turbines.append(o_dg)
            continue

        f_fault_distance_km = d_fault_context.get("fault_distance_km")
        if f_fault_distance_km is None or f_fault_distance_km <= f_dg_distance_km + 1e-9:
            l_skipped_turbines.append(o_dg)
            continue

        l_valid_turbine_items.append({
            "object": o_dg, "terminal": o_dg_terminal,
            "distance_km": f_dg_distance_km, "fault_context": d_fault_context,
        })

    d_summary["turbines_skipped_count"] = len(get_unique_objects(l_skipped_turbines))
    d_summary["turbines_skipped_id"] = get_object_id_string(l_skipped_turbines)

    if not l_valid_turbine_items: return d_summary

    d_summary["turbines_considered_count"] = len(l_valid_turbine_items)
    d_summary["turbines_considered_id"] = get_object_id_string([i["object"] for i in l_valid_turbine_items])

    # Fixed line: Uses Python's built-in sum() correctly now
    d_summary["turbines_total_capacity_mva"] = round(
        sum([get_distributed_generator_capacity_mva(i["object"]) for i in l_valid_turbine_items]), 3)

    l_all_grid_dg = get_all_grid_distributed_generators(o_grid)
    d_original_outserv_states = get_original_outserv_states(l_all_grid_dg)

    l_single_dg_ratios = []
    f_total_correction_r_ohm, f_total_correction_x_ohm = 0.0, 0.0
    f_last_reference_ikss_ka = f_relay_reference_ikss_ka

    try:
        for d_item in l_valid_turbine_items:
            o_dg = d_item["object"]
            activate_only_one_distributed_generator(l_all_grid_dg, o_dg)

            b_success = execute_short_circuit_at_line_fault(
                o_project, d_item["fault_context"]["fault_line"],
                d_item["fault_context"]["fault_percent"], d_item["fault_context"]["local_fault_distance_km"]
            )
            if not b_success: continue

            f_dg_ikss_ka = get_ikss_value(o_dg)
            if f_dg_ikss_ka <= 0.0:
                f_dg_ikss_ka = get_ikss_value(get_pf_attr(o_dg, "bus1"))

            f_reference_ikss_ka, _ = _read_relay_reference_ikss(o_reference_cubicle, o_reference_line,
                                                                s_relay_ikss_line_attr)
            if f_reference_ikss_ka <= 0.0: f_reference_ikss_ka = f_relay_reference_ikss_ka
            if f_reference_ikss_ka <= 0.0: continue

            f_last_reference_ikss_ka = f_reference_ikss_ka
            f_ikss_ratio = f_dg_ikss_ka / f_reference_ikss_ka

            f_impedance_to_fault_r_ohm, f_impedance_to_fault_x_ohm = calculate_path_impedance_between_distances(
                l_path_segments, d_item["distance_km"], d_item["fault_context"]["fault_distance_km"]
            )

            f_total_correction_r_ohm += f_impedance_to_fault_r_ohm * f_ikss_ratio
            f_total_correction_x_ohm += f_impedance_to_fault_x_ohm * f_ikss_ratio
            l_single_dg_ratios.append(f_ikss_ratio)

    finally:
        restore_original_outserv_states(d_original_outserv_states)

    d_summary["relay_reference_ikss_ka"] = round(f_last_reference_ikss_ka, 3)
    d_summary["total_ikss_contribution_ratio"] = round(sum(l_single_dg_ratios), 3)
    d_summary["max_single_ikss_contribution_ratio"] = round(max(l_single_dg_ratios) if l_single_dg_ratios else 0.0, 3)
    d_summary["infeed_correction_r_ohm"] = round(f_total_correction_r_ohm, 3)
    d_summary["infeed_correction_x_ohm"] = round(f_total_correction_x_ohm, 3)

    return d_summary


# =============================================================================
# Case Feature Row Assembly Helpers
# =============================================================================
def get_object_id_string(l_objects):
    return " -> ".join([get_safe_name(obj) for obj in get_unique_objects(l_objects) if obj is not None])


def get_dg_summary_id(d_dg_summary):
    return d_dg_summary.get("names", "")


def get_branch_junction_node_id(d_branch):
    nodes = [term for term in d_branch.get("Branch Terminals", []) if get_terminal_is_junction_node(term)]
    return get_object_id_string(nodes)


def get_branch_distributed_generation_id(d_branch):
    dgs = get_unique_distributed_generators_from_terminals(d_branch.get("Branch Terminals", []))
    return get_object_id_string(dgs)


# =============================================================================
# Row Builder
# =============================================================================
def get_case_feature_row_for_corridor(proj, grid, corr, net, all_corridors, c_idx):
    rb, sb, fl, rc, lines = corr.get("relay_busbar"), corr.get("subsequent_busbar"), corr.get(
        "first_line_section"), corr.get("relay_cubicle"), corr.get("line_sections", [])
    psum = summarize_parallel_protected_corridors(corr, all_corridors)
    vds = filter_valid_downstream_branches_excluding_relay_return(corr, all_corridors)
    nsum = summarize_next_node_downstream_context(vds)
    z2s = select_zone2_downstream_branch_group(corr, all_corridors)
    z2b = z2s.get("selected_branch", build_empty_branch_summary(sb))
    z3b = select_zone3_longest_valid_downstream_branch(corr, all_corridors) or build_empty_branch_summary(sb)
    pgc, _ = count_forward_parallel_branch_groups_for_corridor(corr, all_corridors)
    zmath = calculate_distance_zone_reaches_for_corridor(corr, z2b, psum, all_corridors)
    dgctx = summarize_dg_by_corridor_location(corr, z2b)

    z1t, z2pt = split_protected_corridor_turbines_by_zone_reach(corr, dgctx["protected_corridor"].get("objects", []),
                                                                Config.REACH_GF)
    z2t = get_unique_objects(
        list(z1t) + list(z2pt) + dgctx["subsequent_busbar"].get("objects", []) + dgctx["downstream_branch"].get(
            "objects", []))

    rcub, _, ikss_attr, _ = _get_relay_side_and_cubicle(fl, rb)
    rikss, _ = _read_relay_reference_ikss(rcub or rc, fl, ikss_attr)

    z1inf = calculate_zone_infeed_summary_for_turbines(proj, grid, rb, rcub or rc, fl, ikss_attr, z1t, lines[:],
                                                       zmath.get("zone1_r_reach_ohm", 0.0),
                                                       zmath.get("zone1_x_reach_ohm", 0.0), rikss, "zone1")
    z2inf = calculate_zone_infeed_summary_for_turbines(proj, grid, rb, rcub or rc, fl, ikss_attr, z2t,
                                                       lines + z2b.get("Branch Lines", []),
                                                       zmath.get("zone2_r_reach_ohm", 0.0),
                                                       zmath.get("zone2_x_reach_ohm", 0.0), rikss, "zone2")

    dr = {
        "case_id": f"{c_idx:05d}", "relay_id": get_relay_id_from_terminal_and_line(rb, fl),
        "relay_node_id": get_safe_name(rb), "subsequent_node_id": get_safe_name(sb),
        "protected_corridor_id": corr.get("protected_corridor_id", ""),
        "protected_corridor_length_km": corr.get("total_length_km", 0.0),
        "protected_corridor_r_ohm": corr.get("total_r_ohm", 0.0),
        "protected_corridor_x_ohm": corr.get("total_x_ohm", 0.0), "corridor_hop_count": corr.get("hop_count", 0),
        "line_is_in_service": 1,
        "protected_corridor_is_parallel": get_boolean_value(bool(psum.get("is_parallel"))),
        "protected_corridor_parallel_count": psum.get("parallel_count", 1),
        "protected_corridor_parallel_id": get_parallel_protected_corridor_line_names(corr, all_corridors),
        "next_node_busbar_count": nsum.get("busbar_count", 0),
        "next_node_junction_node_count": nsum.get("junction_node_count", 0),
        "next_node_distributed_generation_count": nsum.get("distributed_generation_count", 0),
        "shortest_downstream_branch_id": z2b.get("Branch ID", ""),
        "shortest_downstream_branch_remote_node_id": z2b.get("Remote Node ID", ""),
        "shortest_downstream_branch_hop_count": int(z2b.get("Hop Count", 0)),
        "shortest_downstream_branch_length_km": z2b.get("Branch Length", 0.0),
        "shortest_downstream_branch_r_ohm": z2b.get("Branch Resistance", 0.0),
        "shortest_downstream_branch_x_ohm": z2b.get("Branch Reactance", 0.0),
        "shortest_downstream_branch_has_parallel": get_boolean_value(z2b.get("Has Parallel", False)),
        "shortest_downstream_branch_parallel_count": int(z2b.get("Parallel Count", 0)),
        "shortest_downstream_branch_parallel_id": z2b.get("Parallel Lines", ""),
        "shortest_downstream_branch_junction_node_count": len(
            [t for t in z2b.get("Branch Terminals", []) if get_terminal_is_junction_node(t)]),
        "shortest_downstream_branch_junction_node_id": " -> ".join(
            [get_safe_name(t) for t in z2b.get("Branch Terminals", []) if get_terminal_is_junction_node(t)]),
        "shortest_downstream_branch_distributed_generation_count": len(
            get_unique_distributed_generators_from_terminals(z2b.get("Branch Terminals", []))),
        "shortest_downstream_branch_distributed_generation_id": " -> ".join([get_safe_name(d) for d in
                                                                             get_unique_distributed_generators_from_terminals(
                                                                                 z2b.get("Branch Terminals", []))]),
        "zone2_branch_selection_method": zmath.get("zone2_branch_selection_method", ""),
        "zone2_selected_branch_id": zmath.get("zone2_downstream_branch_id", ""),
        "zone2_reach_calculation_method": zmath.get("zone2_reach_calculation_method", ""),
        "zone2_impedance_basis": zmath.get("zone2_impedance_basis", ""),
        "parallel_group_count_forward": pgc, "parallel_group_id": zmath.get("zone2_parallel_branch_for_complex_id", ""),
        "zone2_parallel_branch_for_complex_length_km": zmath.get("zone2_parallel_branch_for_complex_length_km", 0.0),
        "zone2_parallel_branch_for_complex_r_ohm": zmath.get("zone2_parallel_branch_for_complex_r_ohm", 0.0),
        "zone2_parallel_branch_for_complex_x_ohm": zmath.get("zone2_parallel_branch_for_complex_x_ohm", 0.0),
        "zone3_branch_selection_method": zmath.get("zone3_branch_selection_method", ""),
        "zone3_selected_branch_id": zmath.get("zone3_downstream_branch_id", ""),
        "longest_downstream_branch_hop_count": len(
            [x for x in zmath.get("zone3_downstream_branch_id", "").split(";") if x.strip()]),
        "longest_downstream_branch_length_km": zmath.get("zone3_downstream_branch_length_km", 0.0),
        "longest_downstream_branch_r_ohm": zmath.get("zone3_downstream_branch_r_ohm", 0.0),
        "longest_downstream_branch_x_ohm": zmath.get("zone3_downstream_branch_x_ohm", 0.0),
        "relay_busbar_distributed_generation_count": dgctx["relay_busbar"].get("count", 0),
        "relay_busbar_distributed_generation_id": dgctx["relay_busbar"].get("names", ""),
        "relay_busbar_distributed_generation_capacity_mva": dgctx["relay_busbar"].get("capacity_mva", 0.0),
        "protected_corridor_distributed_generation_count": dgctx["protected_corridor"].get("count", 0),
        "protected_corridor_distributed_generation_id": dgctx["protected_corridor"].get("names", ""),
        "protected_corridor_distributed_generation_capacity_mva": dgctx["protected_corridor"].get("capacity_mva", 0.0),
        "subsequent_busbar_distributed_generation_count": dgctx["subsequent_busbar"].get("count", 0),
        "subsequent_busbar_distributed_generation_id": dgctx["subsequent_busbar"].get("names", ""),
        "subsequent_busbar_distributed_generation_capacity_mva": dgctx["subsequent_busbar"].get("capacity_mva", 0.0),
        "downstream_branch_distributed_generation_count": dgctx["downstream_branch"].get("count", 0),
        "downstream_branch_distributed_generation_id": dgctx["downstream_branch"].get("names", ""),
        "downstream_branch_distributed_generation_capacity_mva": dgctx["downstream_branch"].get("capacity_mva", 0.0),
        "remote_busbar_distributed_generation_count": dgctx["remote_busbar"].get("count", 0),
        "remote_busbar_distributed_generation_id": dgctx["remote_busbar"].get("names", ""),
        "remote_busbar_distributed_generation_capacity_mva": dgctx["remote_busbar"].get("capacity_mva", 0.0),
        "zone1_turbines_candidate_count": z1inf.get("turbines_candidate_count", 0),
        "zone1_turbines_candidate_id": z1inf.get("turbines_candidate_id", ""),
        "zone1_turbines_considered_count": z1inf.get("turbines_considered_count", 0),
        "zone1_turbines_considered_id": z1inf.get("turbines_considered_id", ""),
        "zone1_turbines_skipped_count": z1inf.get("turbines_skipped_count", 0),
        "zone1_turbines_skipped_id": z1inf.get("turbines_skipped_id", ""),
        "zone1_turbines_total_capacity_mva": z1inf.get("turbines_total_capacity_mva", 0.0),
        "zone1_total_ikss_contribution_ratio": z1inf.get("total_ikss_contribution_ratio", 0.0),
        "zone1_max_single_ikss_contribution_ratio": z1inf.get("max_single_ikss_contribution_ratio", 0.0),
        "zone2_turbines_candidate_count": z2inf.get("turbines_candidate_count", 0),
        "zone2_turbines_candidate_id": z2inf.get("turbines_candidate_id", ""),
        "zone2_turbines_considered_count": z2inf.get("turbines_considered_count", 0),
        "zone2_turbines_considered_id": z2inf.get("turbines_considered_id", ""),
        "zone2_turbines_skipped_count": z2inf.get("turbines_skipped_count", 0),
        "zone2_turbines_skipped_id": z2inf.get("turbines_skipped_id", ""),
        "zone2_turbines_total_capacity_mva": z2inf.get("turbines_total_capacity_mva", 0.0),
        "zone2_total_ikss_contribution_ratio": z2inf.get("total_ikss_contribution_ratio", 0.0),
        "zone2_max_single_ikss_contribution_ratio": z2inf.get("max_single_ikss_contribution_ratio", 0.0),
        "base_zone1_r_reach_ohm": zmath.get("zone1_r_reach_ohm", 0.0),
        "base_zone1_x_reach_ohm": zmath.get("zone1_x_reach_ohm", 0.0),
        "zone1_infeed_correction_r_ohm": z1inf.get("infeed_correction_r_ohm", 0.0),
        "zone1_infeed_correction_x_ohm": z1inf.get("infeed_correction_x_ohm", 0.0),
        "target_zone1_r_reach_ohm": round(
            zmath.get("zone1_r_reach_ohm", 0.0) + z1inf.get("infeed_correction_r_ohm", 0.0), 3),
        "target_zone1_x_reach_ohm": round(
            zmath.get("zone1_x_reach_ohm", 0.0) + z1inf.get("infeed_correction_x_ohm", 0.0), 3),
        "base_zone2_r_reach_ohm": zmath.get("zone2_r_reach_ohm", 0.0),
        "base_zone2_x_reach_ohm": zmath.get("zone2_x_reach_ohm", 0.0),
        "zone2_infeed_correction_r_ohm": z2inf.get("infeed_correction_r_ohm", 0.0),
        "zone2_infeed_correction_x_ohm": z2inf.get("infeed_correction_x_ohm", 0.0),
        "target_zone2_r_reach_ohm": round(
            zmath.get("zone2_r_reach_ohm", 0.0) + z2inf.get("infeed_correction_r_ohm", 0.0), 3),
        "target_zone2_x_reach_ohm": round(
            zmath.get("zone2_x_reach_ohm", 0.0) + z2inf.get("infeed_correction_x_ohm", 0.0), 3),
        "base_zone3_r_reach_ohm": zmath.get("zone3_r_reach_ohm", 0.0),
        "base_zone3_x_reach_ohm": zmath.get("zone3_x_reach_ohm", 0.0),
        "target_zone3_r_reach_ohm": zmath.get("zone3_r_reach_ohm", 0.0),
        "target_zone3_x_reach_ohm": zmath.get("zone3_x_reach_ohm", 0.0)
    }
    return {c: dr.get(c, None) for c in l_case_feature_columns}


# =============================================================================
# Validation & Matrix Export
# =============================================================================
def validate_exports(df):
    missing = [c for c in l_case_feature_columns if
               c not in df.columns or df[c].isna().all() or df[c].astype(str).str.strip().eq("").all()]
    if missing: logger.warning(f"Schema warning: {len(missing)} empty columns -> {missing}")

    cols = ["base_zone1_r_reach_ohm", "zone1_infeed_correction_r_ohm", "target_zone1_r_reach_ohm",
            "base_zone2_r_reach_ohm", "zone2_infeed_correction_r_ohm", "target_zone2_r_reach_ohm"]
    if all(c in df.columns for c in cols):
        dc = df[cols].fillna(0.0).apply(pd.to_numeric, errors="coerce")
        if ((dc["target_zone1_r_reach_ohm"] - (
                dc["base_zone1_r_reach_ohm"] + dc["zone1_infeed_correction_r_ohm"])).abs() > 1e-5).any():
            logger.warning("Reach correction validation failed!")


def export_unified_corridor_feature_matrix(proj, grid):
    logger.info("Creating unified feature matrix...")
    net = extract_network_once(grid)
    corrs = detect_protected_corridors_once(net)
    corrs = sorted(corrs, key=lambda c: (
    get_safe_name(c.get("relay_busbar")), get_safe_name(c.get("subsequent_busbar")),
    get_branch_line_names(c.get("line_sections", []))))

    rows = []
    for i, c in enumerate(corrs, 1):
        logger.info(
            f"Processing {i:03d}: {get_safe_name(c.get('relay_busbar'))} -> {get_safe_name(c.get('subsequent_busbar'))}")
        rows.append(get_case_feature_row_for_corridor(proj, grid, c, net, corrs, i))

    df = pd.DataFrame(rows, columns=l_case_feature_columns)
    validate_exports(df)
    df.to_csv(OUTPUT_DIR / "case_feature_matrix_unified.csv", index=False)
    return df


def export_ml_ready_feature_matrix_from_dataframe(df, outdir, bname):
    targets = ["target_zone1_r_reach_ohm", "target_zone1_x_reach_ohm", "target_zone2_r_reach_ohm",
               "target_zone2_x_reach_ohm", "target_zone3_r_reach_ohm", "target_zone3_x_reach_ohm"]
    drops = ["zone1_infeed_correction_r_ohm", "zone1_infeed_correction_x_ohm", "zone2_infeed_correction_r_ohm",
             "zone2_infeed_correction_x_ohm"]
    tcols = ["scenario_id", "case_uid", "case_id", "switch_state_short_id", "switch_state_config_id",
             "switch_state_row_index", "relay_id", "relay_node_id", "protected_corridor_id", "subsequent_node_id"]

    df[[c for c in tcols if c in df.columns]].to_csv(outdir / f"{bname}_trace_index.csv", index=False)

    dml = df.drop(columns=[c for c in df.columns if
                           c in drops or c in tcols or c.endswith("_id") or c.endswith("_method") or c.endswith(
                               "_basis")], errors="ignore").copy()
    for c in dml.columns: dml[c] = pd.to_numeric(dml[c], errors="coerce")
    dml.dropna(axis=1, how='all', inplace=True)

    feats = [c for c in dml.columns if c not in targets]
    dml[feats] = dml[feats].fillna(0.0)
    dml = dml[feats + [t for t in targets if t in dml.columns]]
    dml.to_csv(outdir / f"{bname}_ml_ready.csv", index=False)
    logger.info(f"ML Ready matrix exported to {bname}_ml_ready.csv")


# =============================================================================
# Switch, Line & Randomization Configs
# =============================================================================
def load_switch_state_dataframe():
    if not Config.ENABLE_SWITCH_STATE_SCENARIOS: return pd.DataFrame([{"ConfigID": "live_grid_state"}])
    if not os.path.exists(SWITCH_STATE_FILE): raise RuntimeError("Switch file missing.")
    df = pd.read_csv(SWITCH_STATE_FILE, sep=None, engine='python')
    sw = [c for c in df.columns if str(c).startswith("switch_")]
    for s in sw: df[s] = pd.to_numeric(df[s], errors="coerce").fillna(1).apply(lambda v: 1 if int(v) == 1 else 0)
    if Config.MAX_SWITCH_STATE_CONFIG_COUNT: df = df.head(Config.MAX_SWITCH_STATE_CONFIG_COUNT)
    return df


def build_cubicle_lookup():
    return {
        c.cimRdfId[0].lstrip("_").strip() if isinstance(c.cimRdfId, list) else str(c.cimRdfId).lstrip("_").strip(): c
        for c in o_pf.GetCalcRelevantObjects("*.StaCubic") or [] if c.cimRdfId}


def get_first_cubicle_switch(c):
    s = c.GetChildren(1, "*.StaSwitch") if c else []
    return s[0] if s else None


def apply_switch_state(row, lookup, cfg):
    app, ms = 0, 0
    for c, v in row.items():
        if not str(c).startswith("switch_"): continue
        rdf = str(c).split("_", 1)[1].lstrip("_").strip()
        sw = get_first_cubicle_switch(lookup.get(rdf))
        if sw:
            sw.on_off = 1 if int(v) == 1 else 0
            app += 1
        else:
            ms += 1
    return {"applied": app, "missing": ms, "config_id": cfg}


def get_switch_state_outserv_controlled_objects(grid):
    objs = []
    for f in ["*.ElmLne", "*.ElmGenstat", "*.ElmPvsys", "*.ElmLod", "*.ElmLodlv", "*.ElmTr2", "*.ElmTr3"]:
        for o in o_pf.GetCalcRelevantObjects(f) or []:
            cubs = [get_pf_attr(o, a) for a in ["bus1", "bus2", "bus3", "bushv", "busmv", "buslv"] if get_pf_attr(o, a)]
            if any(is_object_inside_grid(get_terminal_from_cubicle(c), grid) for c in cubs): objs.append(o)
    return get_unique_objects(objs)


def apply_outserv_for_components_behind_open_switches(objs, orig_states, sid, cfg, scid, log=True):
    forced = 0
    for o in objs:
        if orig_states.get(o, 0) != 0: continue
        cubs = [get_pf_attr(o, a) for a in ["bus1", "bus2", "bus3", "bushv", "busmv", "buslv"] if get_pf_attr(o, a)]
        if any(get_cubicle_switch_closed_state(c) == 0 for c in cubs):
            if safe_set_attribute(o, "outserv", 1): forced += 1
    return {"switch_state_forced_outserv_component_count": forced}, []


def apply_random_line_length_scenario(orig, scid, seed, smin, smax):
    rng, logs = random.Random(seed), []
    for k, v in orig.items():
        if not v["obj"]: continue
        f = rng.uniform(smin, smax)
        nl, nr, nx = v["l"] * f, v["r"] * f, v["x"] * f
        safe_set_attribute(v["obj"], "dline", nl);
        safe_set_attribute(v["obj"], "R1", nr);
        safe_set_attribute(v["obj"], "X1", nx)
        logs.append(
            {"scenario_id": scid, "line_id": get_safe_name(v["obj"]), "scale_factor": f, "original_length_km": v["l"],
             "randomized_length_km": nl})
    return logs


def apply_random_dg_capacity_scenario(orig, scid, seed, smin, smax):
    rng, logs = random.Random(seed), []
    for k, v in orig.items():
        if not v["obj"] or not v["attr"]: continue
        f = rng.uniform(smin, smax)
        nc = v["cap"] * f
        safe_set_attribute(v["obj"], v["attr"], nc)
        logs.append({"scenario_id": scid, "dg_id": get_safe_name(v["obj"]), "scale_factor": f,
                     "original_capacity_mva": v["cap"], "randomized_capacity_mva": nc})
    return logs


def export_randomized_line_length_feature_matrix(proj, grid):
    logger.info("Creating randomized grid-scenario dataset...")
    lines = [l for l in o_pf.GetCalcRelevantObjects("*.ElmLne") or [] if
             get_line_is_available_for_topology(l) and get_line_length_value(l) > 0]
    dgs = get_all_grid_distributed_generators(grid)

    orig_lines = {get_safe_full_name(l): {"obj": l, "l": get_line_length_value(l), "r": get_line_impedance(l)[0],
                                          "x": get_line_impedance(l)[1]} for l in lines}
    orig_dgs = {}
    for dg in dgs:
        for a in ["sgn", "Sn", "snom", "Srated"]:
            val = get_pf_attr(dg, a, None, float)
            if val is not None:
                orig_dgs[get_safe_full_name(dg)] = {"obj": dg, "attr": a, "cap": val};
                break

    df_sw = load_switch_state_dataframe()
    cub_lookup = build_cubicle_lookup()
    orig_sw = {rdf: {"sw": get_first_cubicle_switch(c), "v": int(get_first_cubicle_switch(c).on_off)} for rdf, c in
               cub_lookup.items() if get_first_cubicle_switch(c)}
    out_objs = get_switch_state_outserv_controlled_objects(grid)
    orig_out = {o: get_pf_attr(o, "outserv", 0, int) for o in out_objs}

    all_dfs, log_l, log_d = [], [], []

    try:
        for idx, row in df_sw.iterrows():
            cfg = str(row.get("ConfigID", f"switch_{idx + 1:04d}"))
            sid = f"ss_{idx + 1:04d}"
            logger.info(f"--- Applying Switch State: {sid} ({cfg}) ---")

            if Config.INCLUDE_ORIGINAL_BASE_CASE:
                for v in orig_lines.values(): safe_set_attribute(v["obj"], "dline", v["l"]); safe_set_attribute(
                    v["obj"], "R1", v["r"]); safe_set_attribute(v["obj"], "X1", v["x"])
                for v in orig_dgs.values(): safe_set_attribute(v["obj"], v["attr"], v["cap"])
                for o, v in orig_out.items(): safe_set_attribute(o, "outserv", v)
                for v in orig_sw.values(): v["sw"].on_off = v["v"]

                sumsw = apply_switch_state(row, cub_lookup, cfg)
                apply_outserv_for_components_behind_open_switches(out_objs, orig_out, sid, cfg, "base_0000", False)
                dfb = export_unified_corridor_feature_matrix(proj, grid)
                dfb.insert(0, "switch_state_short_id", sid);
                dfb.insert(1, "scenario_id", "base_0000")
                all_dfs.append(dfb)

            for sc in range(1, Config.RANDOMIZED_SCENARIO_COUNT + 1):
                scid = f"rand_{sc:04d}"
                lseed = Config.RANDOM_SEED_BASE + sc + (idx * Config.RANDOMIZED_SCENARIO_COUNT)
                dseed = lseed + Config.DG_CAPACITY_RANDOM_SEED_OFFSET

                for v in orig_lines.values(): safe_set_attribute(v["obj"], "dline", v["l"]); safe_set_attribute(
                    v["obj"], "R1", v["r"]); safe_set_attribute(v["obj"], "X1", v["x"])
                for v in orig_dgs.values(): safe_set_attribute(v["obj"], v["attr"], v["cap"])
                for o, v in orig_out.items(): safe_set_attribute(o, "outserv", v)
                for v in orig_sw.values(): v["sw"].on_off = v["v"]

                apply_switch_state(row, cub_lookup, cfg)
                apply_outserv_for_components_behind_open_switches(out_objs, orig_out, sid, cfg, scid, False)

                log_l.extend(apply_random_line_length_scenario(orig_lines, scid, lseed, Config.LINE_LENGTH_SCALE_MIN,
                                                               Config.LINE_LENGTH_SCALE_MAX))
                if Config.ENABLE_DG_CAPACITY_RANDOMIZATION:
                    log_d.extend(apply_random_dg_capacity_scenario(orig_dgs, scid, dseed, Config.DG_CAPACITY_SCALE_MIN,
                                                                   Config.DG_CAPACITY_SCALE_MAX))

                dfs = export_unified_corridor_feature_matrix(proj, grid)
                dfs.insert(0, "switch_state_short_id", sid);
                dfs.insert(1, "scenario_id", scid)
                all_dfs.append(dfs)

    finally:
        for v in orig_lines.values(): safe_set_attribute(v["obj"], "dline", v["l"]); safe_set_attribute(v["obj"], "R1",
                                                                                                        v[
                                                                                                            "r"]); safe_set_attribute(
            v["obj"], "X1", v["x"])
        for v in orig_dgs.values(): safe_set_attribute(v["obj"], v["attr"], v["cap"])
        for o, v in orig_out.items(): safe_set_attribute(o, "outserv", v)
        for v in orig_sw.values(): v["sw"].on_off = v["v"]

    if not all_dfs: raise RuntimeError("No scenarios created.")
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(OUTPUT_DIR / "case_feature_matrix_randomized_grid_scenarios.csv", index=False)
    export_ml_ready_feature_matrix_from_dataframe(df_all, OUTPUT_DIR, "case_feature_matrix_randomized_grid_scenarios")

    if log_l: pd.DataFrame(log_l).to_csv(OUTPUT_DIR / "line_length_randomization_log.csv", index=False)
    if log_d: pd.DataFrame(log_d).to_csv(OUTPUT_DIR / "dg_capacity_randomization_log.csv", index=False)

    return df_all


# =============================================================================
# Pipeline Execution
# =============================================================================
def main():
    global o_pf
    logger.info("=" * 80)
    logger.info(f"{Config.DATASET_VERSION} - Supervisor Ready ML Feature Matrix Generator")
    logger.info("=" * 80)
    try:
        o_pf = connect_to_powerfactory()
        proj = activate_powerfactory_project(o_pf, Config.PROJECT_NAME)
        grid = get_target_grid(proj, Config.GRID_NAME)
        if Config.ENABLE_LINE_RANDOMIZATION:
            df = export_randomized_line_length_feature_matrix(proj, grid)
        else:
            df = export_unified_corridor_feature_matrix(proj, grid)
        logger.info("✅ Dataset generation completed successfully.")
        return df
    except Exception as e:
        # Catch any unexpected errors and log the full traceback to the file
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()