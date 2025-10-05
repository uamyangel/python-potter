"""Route node representation and graph structure."""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict
from threading import Lock
from ..global_defs import ObjIdx, CostType, NodeType, IntentCode


@dataclass
class RouteNode:
    """Represents a routing node in the FPGA routing graph."""

    id: int
    tile_id: int
    tile_x: int
    tile_y: int
    wire_id: int
    node_type: NodeType
    intent_code: IntentCode
    base_cost: CostType
    length: int
    is_pinbounce: bool = False
    is_laguna: bool = False
    is_preserved: bool = False
    is_accessible_wire: bool = True

    # Routing-related attributes
    children: List['RouteNode'] = field(default_factory=list)
    parents: List['RouteNode'] = field(default_factory=list)

    # Runtime attributes for routing - use dict for per-net connection counting
    users_connection_counts: Dict[int, int] = field(default_factory=dict)  # net_id -> count
    present_congestion_cost: float = 1.0
    historical_congestion_cost: float = 1.0
    capacity: int = 1

    # Batch stamp for updates
    need_update_batch_stamp: int = -1

    # REMOVED: Lock for thread-safe operations (GIL makes it redundant + adds overhead)
    # With Python's GIL, dict operations on separate keys are effectively atomic
    # For thread-local routing, each thread uses separate scratch arrays

    @property
    def occupancy(self) -> int:
        """Get current occupancy (number of nets using this node)."""
        return len(self.users_connection_counts)

    @property
    def is_overused(self) -> bool:
        """Check if node is overused."""
        return self.occupancy > self.capacity

    @property
    def congestion(self) -> int:
        """Get congestion level."""
        return max(0, self.occupancy - self.capacity)

    def count_connections_of_user(self, net_id: int) -> int:
        """Count how many connections of a net use this node."""
        return self.users_connection_counts.get(net_id, 0)

    def increment_user(self, net_id: int) -> bool:
        """
        Increment connection count for a net.
        Returns True if this is the first connection from this net.
        NO LOCK: GIL + thread-local routing makes explicit locks unnecessary.
        """
        if net_id not in self.users_connection_counts:
            self.users_connection_counts[net_id] = 1
            return True
        else:
            self.users_connection_counts[net_id] += 1
            return False

    def decrement_user(self, net_id: int) -> bool:
        """
        Decrement connection count for a net.
        Returns True if this was the last connection from this net.
        NO LOCK: GIL provides sufficient atomicity for dict operations.
        """
        if net_id in self.users_connection_counts:
            self.users_connection_counts[net_id] -= 1
            if self.users_connection_counts[net_id] <= 0:
                del self.users_connection_counts[net_id]
                return True
        return False

    def clear_users(self) -> None:
        """Clear all users."""
        self.users_connection_counts.clear()

    def update_present_congestion_cost(self, pres_fac: float) -> None:
        """Update present congestion cost based on occupancy."""
        occ = self.occupancy
        if occ < self.capacity:
            self.present_congestion_cost = 1.0
        else:
            self.present_congestion_cost = 1.0 + (occ - self.capacity + 1) * pres_fac


@dataclass
class Connection:
    """Represents a source-sink connection to be routed."""

    id: int
    net_id: int
    source_node: RouteNode
    sink_node: RouteNode
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    center_x: float
    center_y: float
    hpwl: float  # Half-perimeter wirelength
    is_indirect: bool = True

    # Routing result
    route_nodes: List[RouteNode] = field(default_factory=list)
    is_routed: bool = False

    # Optional pre-expanded bounding box for routing (for consistent search space)
    x_min_bb: Optional[int] = None
    x_max_bb: Optional[int] = None
    y_min_bb: Optional[int] = None
    y_max_bb: Optional[int] = None

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass
class Net:
    """Represents a signal net with multiple pins."""

    id: int
    name: str
    connections: List[Connection] = field(default_factory=list)
    x_min: int = 0
    x_max: int = 0
    y_min: int = 0
    y_max: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    hpwl: float = 0.0

    # User tracking for sharing calculations
    users_connection_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def num_connections(self) -> int:
        return len(self.connections)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def count_connections_of_user(self, rnode: 'RouteNode') -> int:
        """Count connections of net using this rnode."""
        return self.users_connection_counts.get(rnode.id, 0)

    def increment_user(self, rnode: 'RouteNode') -> bool:
        """Increment user count. Returns True if new user."""
        if rnode.id not in self.users_connection_counts:
            self.users_connection_counts[rnode.id] = 1
            return True
        else:
            self.users_connection_counts[rnode.id] += 1
            return False

    def decrement_user(self, rnode: 'RouteNode') -> bool:
        """Decrement user count. Returns True if last user removed."""
        if rnode.id in self.users_connection_counts:
            self.users_connection_counts[rnode.id] -= 1
            if self.users_connection_counts[rnode.id] <= 0:
                del self.users_connection_counts[rnode.id]
                return True
        return False


@dataclass
class RouteResult:
    """Stores routing result for a connection."""

    connection_id: int
    net_id: int
    nodes: List[RouteNode] = field(default_factory=list)
    wirelength: int = 0
    success: bool = False
