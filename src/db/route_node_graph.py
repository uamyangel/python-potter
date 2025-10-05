"""Routing graph representation and management."""

from typing import Dict, List, Set
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
