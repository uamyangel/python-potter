"""Routing graph representation and management."""

from typing import Dict, List, Set
import numpy as np
from .route_node import RouteNode


class RouteNodeGraph:
    """Manages the FPGA routing resource graph."""

    def __init__(self):
        self.nodes: List[RouteNode] = []
        self.node_map: Dict[int, RouteNode] = {}  # id -> RouteNode
        self.num_nodes: int = 0
        self.num_edges: int = 0
        # Optional CSR adjacency for faster iteration
        self.csr_indptr: List[int] = []
        self.csr_indices: List[int] = []
        self._has_csr: bool = False

        # NumPy arrays for Numba acceleration (built on demand)
        self.tile_x_arr: np.ndarray = None
        self.tile_y_arr: np.ndarray = None
        self.base_cost_arr: np.ndarray = None
        self.length_arr: np.ndarray = None
        self.node_type_arr: np.ndarray = None
        self.intent_code_arr: np.ndarray = None
        self.capacity_arr: np.ndarray = None
        self.pres_cong_cost_arr: np.ndarray = None
        self.hist_cong_cost_arr: np.ndarray = None
        self.is_accessible_arr: np.ndarray = None
        self._numpy_arrays_built: bool = False

    def add_node(self, node: RouteNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)
        self.node_map[node.id] = node
        self.num_nodes += 1

    def get_node(self, node_id: int) -> RouteNode:
        """Get node by ID."""
        return self.node_map.get(node_id)

    def add_edge(self, from_node: RouteNode, to_node: RouteNode) -> None:
        """Add a directed edge between nodes."""
        from_node.children.append(to_node)
        to_node.parents.append(from_node)
        self.num_edges += 1

    def set_node_children(self) -> None:
        """Initialize node parent-child relationships."""
        # This would be populated from device data
        pass

    def clear_routing_state(self) -> None:
        """Clear all routing state from nodes."""
        for node in self.nodes:
            node.clear_users()
            node.present_congestion_cost = 1.0
            node.historical_congestion_cost = 1.0

    def build_csr(self) -> None:
        """Build CSR adjacency from nodes' children lists."""
        n = len(self.nodes)
        indptr = [0] * (n + 1)
        indices: List[int] = []
        total = 0
        for i, node in enumerate(self.nodes):
            deg = len(node.children)
            total += deg
            indptr[i + 1] = total
        indices = [0] * total
        k = 0
        for node in self.nodes:
            for child in node.children:
                cid = child.id
                indices[k] = cid
                k += 1
        self.csr_indptr = indptr
        self.csr_indices = indices
        self._has_csr = True

    def neighbors(self, node_id: int) -> List[RouteNode]:
        """Get children of node (CSR if available)."""
        if self._has_csr:
            start = self.csr_indptr[node_id]
            end = self.csr_indptr[node_id + 1]
            idxs = self.csr_indices[start:end]
            nodes = self.nodes
            return [nodes[j] for j in idxs]
        else:
            node = self.node_map.get(node_id)
            return node.children if node else []

    def build_numpy_arrays(self) -> None:
        """
        Build NumPy arrays for all node attributes for Numba acceleration.

        This extracts node attributes into contiguous NumPy arrays, allowing
        Numba-compiled functions to access node data with minimal overhead.
        """
        n = len(self.nodes)
        if n == 0:
            return

        self.num_nodes = n

        # Allocate arrays
        self.tile_x_arr = np.empty(n, dtype=np.int32)
        self.tile_y_arr = np.empty(n, dtype=np.int32)
        self.base_cost_arr = np.empty(n, dtype=np.float32)
        self.length_arr = np.empty(n, dtype=np.int16)
        self.node_type_arr = np.empty(n, dtype=np.int16)
        self.intent_code_arr = np.empty(n, dtype=np.int16)
        self.capacity_arr = np.empty(n, dtype=np.int8)
        self.pres_cong_cost_arr = np.empty(n, dtype=np.float32)
        self.hist_cong_cost_arr = np.empty(n, dtype=np.float32)
        self.is_accessible_arr = np.empty(n, dtype=np.bool_)

        # Fill arrays from node objects
        for i, node in enumerate(self.nodes):
            self.tile_x_arr[i] = node.tile_x
            self.tile_y_arr[i] = node.tile_y
            self.base_cost_arr[i] = node.base_cost
            self.length_arr[i] = node.length
            self.node_type_arr[i] = int(node.node_type)
            self.intent_code_arr[i] = int(node.intent_code)
            self.capacity_arr[i] = node.capacity
            self.pres_cong_cost_arr[i] = node.present_congestion_cost
            self.hist_cong_cost_arr[i] = node.historical_congestion_cost
            self.is_accessible_arr[i] = node.is_accessible_wire

        self._numpy_arrays_built = True

    def sync_congestion_costs_to_numpy(self) -> None:
        """
        Sync present/historical congestion costs from RouteNode objects to NumPy arrays.

        Called after cost updates to ensure NumPy arrays reflect current state.
        """
        if not self._numpy_arrays_built:
            return

        for i, node in enumerate(self.nodes):
            self.pres_cong_cost_arr[i] = node.present_congestion_cost
            self.hist_cong_cost_arr[i] = node.historical_congestion_cost
