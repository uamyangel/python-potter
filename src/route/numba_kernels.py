"""
Numba-accelerated kernel functions for A* routing.

This module provides pure numerical functions compiled by Numba for maximum performance.
All complex logic (sharing tracking, queue management) stays in Python to ensure correctness.

FIRST PRINCIPLES:
- No mocking or simplification of logic
- All parameters explicitly passed (no hidden state)
- Real implementations only
"""

import numpy as np
from numba import jit, int32, float32, int16, boolean
from ..global_defs import NodeType


@jit(nopython=True, cache=True)
def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Compute Manhattan distance between two points."""
    return abs(x1 - x2) + abs(y1 - y2)


@jit(nopython=True, cache=True)
def is_in_bbox(
    x: int,
    y: int,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int
) -> boolean:
    """Check if point (x, y) is within bounding box."""
    return x >= x_min and x <= x_max and y >= y_min and y <= y_max


@jit(nopython=True, cache=True)
def compute_node_cost(
    base_cost: float,
    hist_cong_cost: float,
    pres_cong_cost: float,
    sharing_factor: float,
    tile_x: int,
    tile_y: int,
    net_center_x: float,
    net_center_y: float,
    net_hpwl: float,
    net_num_conns: int,
    is_target: boolean
) -> float:
    """
    Compute routing cost for a node (matches C++ Potter exactly).

    Args:
        base_cost: Node base cost
        hist_cong_cost: Historical congestion cost
        pres_cong_cost: Present congestion cost (pre-computed for this node)
        sharing_factor: Sharing factor for this net (1 + sharing_weight * count_source_uses)
        tile_x, tile_y: Node tile coordinates
        net_center_x, net_center_y: Net center for bias calculation
        net_hpwl: Net half-perimeter wirelength
        net_num_conns: Number of connections in net
        is_target: Whether this is the target node

    Returns:
        Total routing cost
    """
    # Bias cost: encourages routing toward net center
    bias_cost = 0.0
    if not is_target and net_hpwl > 0.0:
        bias_cost = (base_cost / max(1, net_num_conns) *
                    (abs(tile_x - net_center_x) + abs(tile_y - net_center_y)) /
                    net_hpwl)

    # Total cost formula (from Potter C++)
    total_cost = (base_cost * hist_cong_cost * pres_cong_cost / sharing_factor +
                 bias_cost)

    return max(0.0, total_cost)


@jit(nopython=True, cache=True)
def check_node_type_accessible(
    node_type: int,
    is_accessible_wire: boolean,
    is_target: boolean
) -> boolean:
    """
    Check node type accessibility (matches C++ Potter logic).

    Args:
        node_type: Node type enum value
        is_accessible_wire: Whether WIRE node is accessible
        is_target: Whether this node is the sink/target

    Returns:
        True if node is accessible given its type
    """
    # NodeType enum values (from global_defs.py)
    WIRE = 0
    PINBOUNCE = 1
    PINFEED_O = 2
    PINFEED_I = 3
    LAGUNA_I = 4
    LAGUNA_O = 5
    SUPER_LONG_LINE = 6

    if node_type == WIRE:
        # Regular wire - check accessibility flag
        return is_accessible_wire

    elif node_type == PINBOUNCE:
        # Pinbounce nodes: not accessible if this is target
        # (matches C++ aStarRoute.cpp:537)
        return not is_target

    elif node_type == PINFEED_I:
        # Input pinfeed: accessible (full logic in Python handles target check)
        # Simple accessibility here
        return True

    elif node_type == PINFEED_O:
        # Output pinfeed: accessible
        return True

    elif node_type == LAGUNA_I or node_type == LAGUNA_O:
        # Laguna tiles: allow for now
        return True

    elif node_type == SUPER_LONG_LINE:
        # Super long lines: allow (cost will penalize if necessary)
        return True

    else:
        # Unknown type: allow but log warning in Python layer
        return True


@jit(nopython=True, cache=True)
def batch_evaluate_children(
    # Children to evaluate
    children_ids: np.ndarray,  # int32[num_children]
    num_children: int,

    # Node attribute arrays
    tile_x: np.ndarray,        # int32[num_nodes]
    tile_y: np.ndarray,
    base_cost: np.ndarray,     # float32[num_nodes]
    length: np.ndarray,        # int16[num_nodes]
    node_type: np.ndarray,     # int16[num_nodes]
    pres_cong_cost: np.ndarray, # float32[num_nodes]
    hist_cong_cost: np.ndarray, # float32[num_nodes]
    is_accessible: np.ndarray,  # bool[num_nodes]

    # Sharing factors (computed in Python)
    sharing_factors: np.ndarray, # float32[num_children]

    # Connection bbox
    x_min_bb: int,
    y_min_bb: int,
    x_max_bb: int,
    y_max_bb: int,

    # Net info for bias cost
    net_center_x: float,
    net_center_y: float,
    net_hpwl: float,
    net_num_conns: int,

    # Sink info for heuristic
    sink_id: int,
    sink_x: int,
    sink_y: int,

    # Routing weights
    current_partial_cost: float,
    rnode_cost_weight: float,
    rnode_wl_weight: float,
    est_wl_weight: float,

    # Output arrays (pre-allocated)
    out_valid: np.ndarray,      # bool[num_children]
    out_total_costs: np.ndarray, # float32[num_children]
    out_partial_costs: np.ndarray # float32[num_children]
) -> int:
    """
    Batch evaluate all children for A* expansion (Numba-accelerated).

    This is the HOT PATH - all numerical computations done here for speed.
    Complex logic (sharing tracking, queue management) stays in Python.

    Returns:
        Number of valid children
    """
    num_valid = 0

    for i in range(num_children):
        child_id = children_ids[i]

        # Default: invalid
        out_valid[i] = False

        # Check if child is sink (handle in Python, just mark valid here)
        if child_id == sink_id:
            out_valid[i] = True
            out_total_costs[i] = current_partial_cost
            out_partial_costs[i] = current_partial_cost
            num_valid += 1
            continue

        # 1) Bbox accessibility check
        child_x = tile_x[child_id]
        child_y = tile_y[child_id]
        if not is_in_bbox(child_x, child_y, x_min_bb, y_min_bb, x_max_bb, y_max_bb):
            continue

        # 2) Node type accessibility
        child_type = node_type[child_id]
        child_is_accessible = is_accessible[child_id]
        is_target = False  # Sink is handled above

        if not check_node_type_accessible(child_type, child_is_accessible, is_target):
            continue

        # 3) Compute cost
        sharing_fac = sharing_factors[i]
        if sharing_fac <= 0.0:
            sharing_fac = 1.0  # Safety check

        node_cost = compute_node_cost(
            base_cost[child_id],
            hist_cong_cost[child_id],
            pres_cong_cost[child_id],
            sharing_fac,
            child_x,
            child_y,
            net_center_x,
            net_center_y,
            net_hpwl,
            net_num_conns,
            is_target
        )

        # 4) Partial path cost (accumulated cost so far)
        child_len = float(length[child_id])
        new_partial = (current_partial_cost +
                      rnode_cost_weight * node_cost +
                      rnode_wl_weight * child_len / sharing_fac)

        # 5) Heuristic (Manhattan distance to sink)
        dist_to_sink = manhattan_distance(child_x, child_y, sink_x, sink_y)

        # 6) Total cost (f = g + h)
        new_total = new_partial + est_wl_weight * float(dist_to_sink) / sharing_fac

        # Mark as valid
        out_valid[i] = True
        out_total_costs[i] = new_total
        out_partial_costs[i] = new_partial
        num_valid += 1

    return num_valid


@jit(nopython=True, cache=True)
def update_congestion_costs_batch(
    node_ids: np.ndarray,       # int32[num_nodes_to_update]
    num_nodes_to_update: int,
    occupancy: np.ndarray,      # int32[num_nodes] - computed in Python
    capacity: np.ndarray,       # int8[num_nodes]
    pres_fac: float,
    hist_fac: float,
    # Output arrays (modified in place)
    pres_cong_cost: np.ndarray, # float32[num_nodes]
    hist_cong_cost: np.ndarray  # float32[num_nodes]
):
    """
    Batch update congestion costs for overused nodes (Numba-accelerated).

    This is called after each iteration to update costs for overused/dirty nodes.
    """
    for i in range(num_nodes_to_update):
        node_id = node_ids[i]

        occ = occupancy[node_id]
        cap = capacity[node_id]
        overuse = occ - cap

        if overuse <= 0:
            pres_cong_cost[node_id] = 1.0 + pres_fac
        else:
            pres_cong_cost[node_id] = 1.0 + (overuse + 1) * pres_fac
            hist_cong_cost[node_id] += overuse * hist_fac
