"""Partition tree for parallel routing."""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from ..db.route_node import Net
from ..global_defs import Box


@dataclass
class PartitionBBox(Box):
    """Partition bounding box with additional attributes."""

    id: int = 0
    level: int = 0
    nets: Set[int] = field(default_factory=set)


@dataclass
class PartitionTreeNode:
    """Node in partition tree."""

    bbox: PartitionBBox
    children: List['PartitionTreeNode'] = field(default_factory=list)
    parent: Optional['PartitionTreeNode'] = None
    nets: Set[int] = field(default_factory=set)
    level: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class PartitionTree:
    """Hierarchical partitioning tree for parallel routing."""

    def __init__(self, nets: List[Net], device_bbox: PartitionBBox):
        self.root = PartitionTreeNode(bbox=device_bbox, level=0)
        self.nets = nets
        self.leaves: List[PartitionTreeNode] = []
        self.max_depth = 0
        self.max_depth_limit = 32  # Safety guard to prevent excessive recursion

    def build(self, max_nets_per_partition: int = 1000) -> None:
        """Build partition tree by recursively subdividing device."""
        self._partition_recursive(self.root, max_nets_per_partition)
        self._collect_leaves(self.root)

    def _partition_recursive(self, node: PartitionTreeNode, max_nets: int) -> None:
        """Recursively partition a node."""
        # Depth guard
        if node.level >= self.max_depth_limit:
            return

        # Collect nets intersecting this partition
        for net in self.nets:
            if node.bbox.overlaps(Box(net.x_min, net.y_min, net.x_max, net.y_max)):
                node.nets.add(net.id)

        # Stop if few enough nets
        if len(node.nets) <= max_nets:
            return

        # Stop if region cannot be subdivided further (degenerate box)
        if node.bbox.width <= 1 and node.bbox.height <= 1:
            return

        # Split into children (simple bisection)
        mid_x = (node.bbox.x_min + node.bbox.x_max) // 2
        mid_y = (node.bbox.y_min + node.bbox.y_max) // 2

        # If split does not change bounds, abort to avoid infinite recursion
        if (mid_x <= node.bbox.x_min or mid_x >= node.bbox.x_max or
                mid_y <= node.bbox.y_min or mid_y >= node.bbox.y_max):
            return

        # Create 4 children (quadtree partitioning)
        children_boxes = [
            PartitionBBox(node.bbox.x_min, node.bbox.y_min, mid_x, mid_y),
            PartitionBBox(mid_x, node.bbox.y_min, node.bbox.x_max, mid_y),
            PartitionBBox(node.bbox.x_min, mid_y, mid_x, node.bbox.y_max),
            PartitionBBox(mid_x, mid_y, node.bbox.x_max, node.bbox.y_max),
        ]

        for i, bbox in enumerate(children_boxes):
            child = PartitionTreeNode(
                bbox=bbox,
                parent=node,
                level=node.level + 1
            )
            node.children.append(child)
            self._partition_recursive(child, max_nets)

        self.max_depth = max(self.max_depth, node.level + 1)

    def _collect_leaves(self, node: PartitionTreeNode) -> None:
        """Collect all leaf nodes."""
        if node.is_leaf:
            self.leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child)

    def get_level_nodes(self, level: int) -> List[PartitionTreeNode]:
        """Get all nodes at a specific level."""
        nodes = []
        self._collect_level_nodes(self.root, level, nodes)
        return nodes

    def _collect_level_nodes(
        self,
        node: PartitionTreeNode,
        target_level: int,
        result: List[PartitionTreeNode]
    ) -> None:
        """Helper to collect nodes at a level."""
        if node.level == target_level:
            result.append(node)
        elif node.level < target_level:
            for child in node.children:
                self._collect_level_nodes(child, target_level, result)

    def find_leaf_for_point(self, x: int, y: int) -> Optional[PartitionTreeNode]:
        """Find the leaf node whose bbox contains the point (x, y)."""
        node = self.root
        # If outside root bbox, clamp to root
        if not node.bbox.contains(x, y):
            return self._find_nearest_leaf(node)
        while not node.is_leaf:
            next_node = None
            for child in node.children:
                if child.bbox.contains(x, y):
                    next_node = child
                    break
            if next_node is None:
                # On boundary or missing; pick the first child as fallback
                next_node = node.children[0] if node.children else node
            node = next_node
        return node

    def _find_nearest_leaf(self, node: PartitionTreeNode) -> Optional[PartitionTreeNode]:
        if node.is_leaf:
            return node
        for child in node.children:
            leaf = self._find_nearest_leaf(child)
            if leaf is not None:
                return leaf
        return None
