"""A* routing algorithm implementation."""

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
import numpy as np
from ..db.route_node import Connection, RouteNode, Net
from ..db.database import Database
from ..utils.log import log
from ..global_defs import NodeType

# Import Numba kernels for acceleration
try:
    from .numba_kernels import (
        batch_evaluate_children,
        numba_heap_push,
        numba_heap_pop,
        numba_heap_peek
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    log("Warning: Numba not available, falling back to pure Python (slower)")


@dataclass
class NodeInfo:
    """Information about a node during A* search."""

    prev: Optional[RouteNode] = None
    cost: float = float('inf')
    partial_cost: float = 0.0
    is_visited: int = -1  # Connection unique ID that visited this node
    is_target: int = -1   # Connection unique ID that marked this as target
    occ_change: int = 0   # For stable-first routing

    def __lt__(self, other):
        return self.cost < other.cost

    def write(self, prev: Optional[RouteNode], cost: float, partial_cost: float,
             is_visited: int, is_target: int):
        """Write node info for this search."""
        self.prev = prev
        self.cost = cost
        self.partial_cost = partial_cost
        self.is_visited = is_visited
        self.is_target = is_target


class AStarRouter:
    """A* pathfinding router for a single connection."""

    def __init__(
        self,
        database: Database,
        present_cong_factor: float = 0.5,
        historical_cong_factor: float = 1.0,
        rnode_cost_weight: float = 1.0,
        rnode_wl_weight: float = 0.2,
        est_wl_weight: float = 0.8,
        sharing_weight: float = 1.0,
        use_numba_heap: bool = True  # New: enable Numba heap for 2-3x speedup
    ):
        self.database = database
        self.present_cong_factor = present_cong_factor
        self.historical_cong_factor = historical_cong_factor
        self.rnode_cost_weight = rnode_cost_weight
        self.rnode_wl_weight = rnode_wl_weight
        self.est_wl_weight = est_wl_weight
        self.sharing_weight = sharing_weight
        self.use_numba_heap = use_numba_heap and NUMBA_AVAILABLE

        # Cache graph arrays reference for Numba
        self._use_numba = NUMBA_AVAILABLE and hasattr(database.routing_graph, '_numpy_arrays_built') and database.routing_graph._numpy_arrays_built
        if self._use_numba:
            graph = database.routing_graph
            self._tile_x = graph.tile_x_arr
            self._tile_y = graph.tile_y_arr
            self._base_cost = graph.base_cost_arr
            self._length = graph.length_arr
            self._node_type = graph.node_type_arr
            self._pres_cong_cost = graph.pres_cong_cost_arr
            self._hist_cong_cost = graph.hist_cong_cost_arr
            self._is_accessible = graph.is_accessible_arr  # numpy array for indexing
            # Pre-allocate work arrays for batch evaluation
            self._max_children = 128  # Typical fanout
            self._work_valid = np.empty(self._max_children, dtype=np.bool_)
            self._work_total_costs = np.empty(self._max_children, dtype=np.float32)
            self._work_partial_costs = np.empty(self._max_children, dtype=np.float32)
            self._work_sharing_factors = np.empty(self._max_children, dtype=np.float32)

        # Pre-allocate Numba heap arrays (reused across connections)
        if self.use_numba_heap:
            self._heap_max_size = 200000  # Match max_nodes in route_one_connection
            self._heap_costs = np.empty(self._heap_max_size, dtype=np.float32)
            self._heap_node_ids = np.empty(self._heap_max_size, dtype=np.int32)
            self._heap_size = 0  # Current heap size

    def route_one_connection(
        self,
        connection: Connection,
        node_infos: Union[Dict[int, NodeInfo], 'NodeScratch'],
        connection_unique_id: int,
        tid: int = 0,
        sync: bool = False
    ) -> bool:
        """
        Route a single connection using A* algorithm.

        Args:
            connection: Connection to route
            node_infos: Per-thread node info dictionary
            connection_unique_id: Unique ID for this routing iteration
            tid: Thread ID
            sync: Whether to use synchronous mode (for stable-first)

        Returns:
            True if routing succeeded, False otherwise
        """

        # Clear per-thread scratch to avoid unbounded growth across connections
        if isinstance(node_infos, dict):
            if node_infos:
                node_infos.clear()

        source = connection.source_node
        sink = connection.sink_node
        net = self.database.nets[connection.net_id]

        # Priority queue using cost-based comparison
        # Store (cost, RouteNode) tuples
        pq: List[tuple] = []

        def push(rnode: RouteNode, prev: Optional[RouteNode],
                cost: float, partial_cost: float, is_target: int):
            """Push node onto priority queue."""
            if isinstance(node_infos, NodeScratch):
                nid = rnode.id
                node_infos.prev[nid] = prev.id if prev is not None else -1
                node_infos.cost[nid] = cost
                node_infos.partial[nid] = partial_cost
                node_infos.visited[nid] = connection_unique_id
                if is_target != -1:
                    node_infos.target[nid] = connection_unique_id
            else:
                ninfo = node_infos.get(rnode.id)
                if ninfo is None:
                    ninfo = NodeInfo()
                    node_infos[rnode.id] = ninfo
                ninfo.write(prev, cost, partial_cost, connection_unique_id, is_target)
            heapq.heappush(pq, (cost, rnode.id, rnode))

        # Initialize source
        push(source, None, 0.0, 0.0, -1)

        # Mark sink
        if isinstance(node_infos, NodeScratch):
            node_infos.target[sink.id] = connection_unique_id
        else:
            sink_info = node_infos.get(sink.id)
            if sink_info is None:
                sink_info = NodeInfo()
                node_infos[sink.id] = sink_info
            sink_info.write(None, 0.0, 0.0, -1, connection_unique_id)

        target_rnode: Optional[RouteNode] = None
        nodes_popped = 0
        max_nodes = 200000

        while pq and nodes_popped < max_nodes:
            _, _, rnode = heapq.heappop(pq)
            nodes_popped += 1

            if isinstance(node_infos, NodeScratch):
                ninfo_partial_cost = node_infos.partial[rnode.id]
            else:
                ninfo = node_infos.get(rnode.id)
                if ninfo is None:
                    # Defensive: ensure node info exists for any popped node
                    ninfo = NodeInfo()
                    node_infos[rnode.id] = ninfo
                ninfo_partial_cost = ninfo.partial_cost

            # Check if we reached the sink
            if rnode.id == sink.id:
                target_rnode = rnode
                break

            # Expand children - USE CSR for cache-friendly iteration
            graph = self.database.routing_graph
            if graph._has_csr:
                # CSR: get child node IDs and look up from node_map
                start_idx = graph.csr_indptr[rnode.id]
                end_idx = graph.csr_indptr[rnode.id + 1]
                child_node_ids = graph.csr_indices[start_idx:end_idx]
                
                # NUMBA BATCH EVALUATION - Main acceleration point
                if self._use_numba and len(child_node_ids) > 0:
                    try:
                        # Convert to numpy array for Numba
                        child_ids_array = np.array(child_node_ids, dtype=np.int32)
                        
                        # Batch evaluate using Numba (FAST!)
                        result = self._expand_children_numba(
                            child_ids_array,
                            connection,
                            net,
                            ninfo_partial_cost
                        )
                        
                        if result is not None:
                            valid_ids, total_costs, partial_costs = result
                            
                            # Process all valid children
                            for i in range(len(valid_ids)):
                                child_id = valid_ids[i]
                                child_rnode = graph.node_map.get(child_id)
                                if child_rnode is None:
                                    continue
                                
                                # Check if already visited (Python manages visited set)
                                if isinstance(node_infos, NodeScratch):
                                    if node_infos.visited[child_id] == connection_unique_id:
                                        continue
                                else:
                                    child_info = node_infos.get(child_id)
                                    if child_info and child_info.is_visited == connection_unique_id:
                                        continue
                                
                                # Check if this is the sink (target found!)
                                if child_id == connection.sink_node.id:
                                    target_rnode = child_rnode
                                    # Set predecessor for path reconstruction
                                    if isinstance(node_infos, NodeScratch):
                                        node_infos.prev[child_id] = rnode.id
                                    else:
                                        child_info = node_infos.get(child_id)
                                        if child_info is None:
                                            child_info = NodeInfo()
                                            node_infos[child_id] = child_info
                                        child_info.prev = rnode
                                    break
                                
                                # Push to priority queue
                                push(child_rnode, rnode, total_costs[i], partial_costs[i], -1)
                            
                            # Check if target found
                            if target_rnode is not None:
                                break
                            
                            # Skip Python loop - Numba handled everything
                            continue
                            
                    except Exception as e:
                        # Fallback to Python if Numba fails
                        log(f"WARNING: Numba batch evaluation failed: {e}, falling back to Python")
                
                # Use node_map for lookup (ids may not be sequential) - FALLBACK
                node_map = graph.node_map
                children_iter = [node_map[nid] for nid in child_node_ids if nid in node_map]
            else:
                # No CSR - must use rnode.children (may be empty in worker processes!)
                # SAFETY: In multiprocessing mode, children lists are empty.
                # This should never happen because prepare_database_state requires CSR.
                children_iter = rnode.children
                if not children_iter and hasattr(graph, 'csr_indptr'):
                    # Defensive: If children empty but CSR exists, use CSR
                    start_idx = graph.csr_indptr[rnode.id]
                    end_idx = graph.csr_indptr[rnode.id + 1]
                    child_node_ids = graph.csr_indices[start_idx:end_idx]
                    children_iter = [graph.node_map[nid] for nid in child_node_ids if nid in graph.node_map]

            # Python fallback loop (only runs if Numba disabled or failed)
            for child_rnode in children_iter:
                if isinstance(node_infos, NodeScratch):
                    is_visited = (node_infos.visited[child_rnode.id] == connection_unique_id)
                    is_target = (node_infos.target[child_rnode.id] == connection_unique_id)
                else:
                    child_info = node_infos.get(child_rnode.id)
                    if child_info is None:
                        child_info = NodeInfo()
                        node_infos[child_rnode.id] = child_info
                    is_visited = (child_info.is_visited == connection_unique_id)
                    is_target = (child_info.is_target == connection_unique_id)

                if is_visited:
                    continue

                # Check if child is sink (direct neighbor case)
                if child_rnode.id == sink.id:
                    target_rnode = child_rnode
                    if isinstance(node_infos, NodeScratch):
                        node_infos.prev[child_rnode.id] = rnode.id
                    else:
                        child_info.prev = rnode
                    break

                # Check accessibility
                if self._use_numba:
                    # Use numpy array for Numba acceleration
                    if not self._is_accessible[child_rnode.id]:
                        continue
                else:
                    # Use method call for regular Python
                    if not self._is_accessible_method(child_rnode, connection):
                        continue

                # Check node-type specific accessibility rules
                if not self._check_node_type_accessibility(
                    child_rnode, connection, is_target
                ):
                    continue

                # Calculate costs
                count_source_uses_origin = net.count_connections_of_user(child_rnode)
                count_source_uses = count_source_uses_origin

                occ_change = 0

                sharing_factor = 1.0 + self.sharing_weight * count_source_uses

                node_cost = self._get_node_cost(
                    child_rnode, connection, occ_change,
                    count_source_uses, count_source_uses_origin,
                    sharing_factor, is_target, tid
                )

                # Partial path cost
                new_partial_cost = (ninfo_partial_cost +
                                  self.rnode_cost_weight * node_cost +
                                  self.rnode_wl_weight * child_rnode.length / sharing_factor)

                # Heuristic (Manhattan distance to sink)
                delta_x = abs(child_rnode.tile_x - sink.tile_x)
                delta_y = abs(child_rnode.tile_y - sink.tile_y)
                distance_to_sink = delta_x + delta_y

                # Total cost
                new_total_cost = (new_partial_cost +
                                self.est_wl_weight * distance_to_sink / sharing_factor)

                push(child_rnode, rnode, new_total_cost, new_partial_cost, -1)

            if target_rnode is not None:
                break

        # Check if routing succeeded
        if target_rnode is None:
            return False

        # Save routing solution
        routed = self._save_routing(connection, target_rnode, node_infos, tid)

        if routed and not sync:
            self._update_users_and_present_congestion_cost(connection)

        return routed

    def _is_accessible_method(self, node: RouteNode, connection: Connection) -> bool:
        """
        Check if node is accessible for this connection within a relaxed bbox.
        Allows exploration inside connection bbox with adaptive margin.
        """
        # Prefer precomputed bbox bounds when available (consistent with C++ behavior)
        if connection.x_min_bb is not None:
            return (
                node.tile_x >= connection.x_min_bb and
                node.tile_x <= connection.x_max_bb and
                node.tile_y >= connection.y_min_bb and
                node.tile_y <= connection.y_max_bb
            )
        # Fallback: adaptive margin based on connection size (hpwl proxy)
        w = max(1, connection.width)
        h = max(1, connection.height)
        hpwl = w + h
        margin = max(4, min(30, hpwl // 6))
        return (
            node.tile_x >= connection.x_min - margin and
            node.tile_x <= connection.x_max + margin and
            node.tile_y >= connection.y_min - margin and
            node.tile_y <= connection.y_max + margin
        )

    def _check_node_type_accessibility(
        self,
        node: RouteNode,
        connection: Connection,
        is_target: bool
    ) -> bool:
        """Check node-type specific accessibility rules."""

        if node.node_type == NodeType.WIRE:
            # Regular wire - check if accessible
            if not node.is_accessible_wire:
                return False

        elif node.node_type == NodeType.PINBOUNCE:
            # Pinbounce nodes have special rules
            if is_target:
                return False
            # Additional pinbounce checks would go here

        elif node.node_type == NodeType.PINFEED_I:
            # Input pinfeed has special rules
            # Only accessible for sink pins in most cases
            pass

        elif node.node_type == NodeType.LAGUNA_I:
            # Allow for now; detailed handling can be added if needed
            pass

        elif node.node_type == NodeType.SUPER_LONG_LINE:
            # Allow long lines; cost function will penalize if necessary
            pass

        return True

    def _update_users_and_present_congestion_cost(self, connection: Connection) -> None:
        """Update node usage and congestion costs after routing."""
        net = self.database.nets[connection.net_id]

        for rnode in connection.route_nodes:
            # Update node users
            is_new_user = rnode.increment_user(connection.net_id)

            # Update net's user tracking
            net.increment_user(rnode)

            # Update present congestion cost
            rnode.update_present_congestion_cost(self.present_cong_factor)

    def should_route(self, connection: Connection) -> bool:
        """Check if connection needs to be (re-)routed."""
        if not connection.is_routed:
            return True

        # Check if route is congested
        for rnode in connection.route_nodes:
            if rnode.is_overused:
                return True

        return False

    def _get_node_cost(
        self,
        rnode: RouteNode,
        connection: Connection,
        occ_change: int,
        count_source_uses: int,
        count_source_uses_origin: int,
        sharing_factor: float,
        is_target: bool,
        tid: int
    ) -> float:
        """
        Calculate the cost of using a routing node.

        This includes:
        - Base cost of the node
        - Historical congestion cost
        - Present congestion cost (adjusted for same-net sharing)
        - Bias cost towards net center
        """

        has_same_source_users = (count_source_uses != 0)
        net = self.database.nets[connection.net_id]

        # Calculate present congestion cost
        if has_same_source_users:
            # Node used by same net - reduce congestion penalty
            pre_dec_occ = 1 if (count_source_uses_origin > 0 and count_source_uses == 0) else 0
            pre_inc_occ = 1 if (count_source_uses_origin == 0 and count_source_uses > 0) else 0

            over_occupancy = (rnode.occupancy - pre_dec_occ + pre_inc_occ +
                            occ_change - rnode.capacity)
            present_cong_cost = 1.0 + over_occupancy * self.present_cong_factor
        else:
            present_cong_cost = rnode.present_congestion_cost

        # Bias cost - encourages routing toward net center
        bias_cost = 0.0
        if not is_target and net.hpwl > 0:
            bias_cost = (rnode.base_cost / max(1, net.num_connections) *
                        (abs(rnode.tile_x - net.center_x) +
                         abs(rnode.tile_y - net.center_y)) /
                        net.hpwl)

        # Total cost
        total_cost = (rnode.base_cost *
                     rnode.historical_congestion_cost *
                     present_cong_cost / sharing_factor +
                     bias_cost)

        return max(0.0, total_cost)

    def _save_routing(
        self,
        connection: Connection,
        sink_rnode: RouteNode,
        node_infos: Union[Dict[int, NodeInfo], 'NodeScratch'],
        tid: int
    ) -> bool:
        """
        Save routing solution by backtracing from sink to source.
        """

        route = []
        current = sink_rnode
        watchdog = 0
        max_iterations = 10000

        # Backtrace from sink to source
        while current is not None and watchdog < max_iterations:
            route.append(current)
            if isinstance(node_infos, NodeScratch):
                prev_id = node_infos.prev[current.id]
                if prev_id < 0:
                    if current.id == connection.source_node.id:
                        break
                    else:
                        return False
                prev_node = self.database.routing_graph.node_map.get(prev_id)
                current = prev_node
            else:
                ninfo = node_infos.get(current.id)
                if ninfo is None or ninfo.prev is None:
                    if current.id == connection.source_node.id:
                        break
                    else:
                        # Failed - incomplete path
                        return False
                current = ninfo.prev
            watchdog += 1

        if watchdog >= max_iterations:
            return False

        if len(route) < 2:
            return False

        # Reverse to get source -> sink order
        route.reverse()

        # Save result
        connection.route_nodes = route
        connection.is_routed = True

        return True


    def _expand_children_numba(
        self,
        child_ids: np.ndarray,
        connection: Connection,
        net: Net,
        current_partial_cost: float
    ) -> tuple:
        """
        Batch evaluate children using Numba acceleration.

        FIRST PRINCIPLES: Real sharing_factor calculation (no mocking).

        Args:
            child_ids: Array of child node IDs
            connection: Connection being routed
            net: Net for this connection
            current_partial_cost: Accumulated cost so far

        Returns:
            (valid_children_ids, total_costs, partial_costs) arrays
        """
        if not self._use_numba:
            return None  # Fall back to Python

        num_children = len(child_ids)
        if num_children == 0:
            return (np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.float32))

        # Debug: Log first few calls to verify Numba is active
        if not hasattr(self, '_numba_call_count'):
            self._numba_call_count = 0
        self._numba_call_count += 1
        if self._numba_call_count <= 3:
            log(f"DEBUG: Numba batch evaluation called (#{self._numba_call_count}) with {num_children} children")

        # Resize work arrays if needed
        if num_children > len(self._work_valid):
            self._work_valid = np.empty(num_children, dtype=np.bool_)
            self._work_total_costs = np.empty(num_children, dtype=np.float32)
            self._work_partial_costs = np.empty(num_children, dtype=np.float32)
            self._work_sharing_factors = np.empty(num_children, dtype=np.float32)

        # CRITICAL: Compute REAL sharing factors for each child (no mock/simplification)
        for i, child_id in enumerate(child_ids):
            count_source_uses = net.count_connections_of_user(self.database.routing_graph.node_map[child_id])
            self._work_sharing_factors[i] = 1.0 + self.sharing_weight * count_source_uses

        # Get bbox (prefer precomputed)
        if connection.x_min_bb is not None:
            x_min, x_max = connection.x_min_bb, connection.x_max_bb
            y_min, y_max = connection.y_min_bb, connection.y_max_bb
        else:
            margin_x, margin_y = 3, 15
            x_min = connection.x_min - margin_x
            x_max = connection.x_max + margin_x
            y_min = connection.y_min - margin_y
            y_max = connection.y_max + margin_y

        # Call Numba-accelerated batch evaluation
        num_valid = batch_evaluate_children(
            child_ids,
            num_children,
            self._tile_x,
            self._tile_y,
            self._base_cost,
            self._length,
            self._node_type,
            self._pres_cong_cost,
            self._hist_cong_cost,
            self._is_accessible,
            self._work_sharing_factors,
            x_min, y_min, x_max, y_max,
            net.center_x, net.center_y, net.hpwl, net.num_connections,
            connection.sink_node.id,
            connection.sink_node.tile_x,
            connection.sink_node.tile_y,
            current_partial_cost,
            self.rnode_cost_weight,
            self.rnode_wl_weight,
            self.est_wl_weight,
            self._work_valid,
            self._work_total_costs,
            self._work_partial_costs
        )

        if num_valid == 0:
            return (np.array([], dtype=np.int32), np.array([], dtype=np.float32), np.array([], dtype=np.float32))

        # Extract valid results
        valid_mask = self._work_valid[:num_children]
        valid_ids = child_ids[valid_mask]
        valid_total_costs = self._work_total_costs[:num_children][valid_mask]
        valid_partial_costs = self._work_partial_costs[:num_children][valid_mask]

        return (valid_ids, valid_total_costs, valid_partial_costs)


class NodeScratch:
    """Array-backed scratch space for A* to reduce Python overhead.

    Uses visited/target stamps so arrays need not be cleared per-connection.
    """

    def __init__(self, num_nodes: int):
        self.prev = np.full(num_nodes, -1, dtype=np.int32)
        self.cost = np.full(num_nodes, np.inf, dtype=np.float64)
        self.partial = np.zeros(num_nodes, dtype=np.float64)
        self.visited = np.full(num_nodes, -1, dtype=np.int64)
        self.target = np.full(num_nodes, -1, dtype=np.int64)

    # NodeScratch holds no algorithmic methods; A* methods live on AStarRouter
