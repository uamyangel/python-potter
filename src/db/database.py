"""Main database class coordinating device and netlist."""

from typing import List
from .device import Device
from .netlist import Netlist
from .route_node import Connection, Net, RouteResult
from .route_node_graph import RouteNodeGraph
from ..global_defs import Box
from ..utils.log import log


class Database:
    """Main database coordinating device, netlist, and routing graph."""

    def __init__(self):
        self.layout = Box(x_min=0, y_min=0, x_max=108, y_max=300)
        self.routing_graph = RouteNodeGraph()
        self.device = Device(self.routing_graph)

        # Data structures
        self.nets: List[Net] = []
        self.indirect_connections: List[Connection] = []
        self.direct_connections: List[Connection] = []
        self.preserved_nodes: List[bool] = []

        self.netlist = Netlist(
            self.device,
            self.nets,
            self.indirect_connections,
            self.direct_connections,
            self.preserved_nodes,
            self.routing_graph,
            self.layout
        )

        # Statistics
        self.num_nodes = 0
        self.num_nodes_in_rrg = 0
        self.num_edges = 0
        self.num_edges_in_rrg = 0
        self.num_conns = 0
        self.num_nets = 0
        self.preserved_num = 0

        # Configuration
        self.num_thread = 16
        self.use_rw = False
        self.device_cache_mode = "off"
        self.input_name = ""

    def set_num_thread(self, n: int) -> None:
        """Set number of threads."""
        self.num_thread = n
        self.netlist.num_thread = n

    def get_num_thread(self) -> int:
        """Get number of threads."""
        return self.num_thread

    def set_device_cache_mode(self, mode: str) -> None:
        """Set device cache mode: 'off' or 'light'."""
        self.device_cache_mode = mode

    def read_device(self, device_name: str) -> None:
        """Read device file."""
        self.device.read(device_name, cache_mode=self.device_cache_mode)
        self.num_nodes = self.routing_graph.num_nodes
        self.num_edges = self.routing_graph.num_edges
        # Build CSR adjacency if not built by device
        try:
            if not getattr(self.routing_graph, '_has_csr', False):
                self.routing_graph.build_csr()
        except Exception:
            pass
        # Update layout bounds based on device tiles (col=x, row=y)
        try:
            if hasattr(self.device, 'tiles') and self.device.tiles:
                xs = [t.get('col', 0) for t in self.device.tiles]
                ys = [t.get('row', 0) for t in self.device.tiles]
                self.layout = Box(
                    x_min=min(xs) if xs else 0,
                    y_min=min(ys) if ys else 0,
                    x_max=max(xs) if xs else 0,
                    y_max=max(ys) if ys else 0,
                )
        except Exception:
            pass

        # Build NumPy arrays for Numba acceleration
        log("Building NumPy arrays for Numba acceleration...")
        try:
            self.routing_graph.build_numpy_arrays()
            log(f"  NumPy arrays built: {self.routing_graph.num_nodes:,} nodes")
        except Exception as e:
            log(f"  Warning: Failed to build NumPy arrays: {e}")
            log("  Routing will fall back to pure Python (slower)")


    def read_netlist(self, netlist_name: str) -> None:
        """Read netlist file."""
        self.input_name = netlist_name
        self.netlist.read(netlist_name)
        self.num_nets = len(self.nets)
        self.num_conns = len(self.indirect_connections) + len(self.direct_connections)

    def write_netlist(self, netlist_name: str, routing_results: List[RouteResult]) -> None:
        """Write routed netlist."""
        self.netlist.write(netlist_name, routing_results)

    def set_route_node_children(self) -> None:
        """Initialize routing node parent-child relationships."""
        self.routing_graph.set_node_children()

    def reduce_route_node(self) -> None:
        """Reduce routing graph size by removing unnecessary nodes."""
        log("Route node reduction not yet implemented")

    def print_statistic(self) -> None:
        """Print database statistics."""
        log("=" * 60)
        log("Database Statistics:")
        log(f"  Device: {self.device.name}")
        log(f"  Nodes: {self.num_nodes}")
        log(f"  Edges: {self.num_edges}")
        log(f"  Nets: {self.num_nets}")
        log(f"  Connections: {self.num_conns}")
        log(f"    Indirect: {len(self.indirect_connections)}")
        log(f"    Direct: {len(self.direct_connections)}")
        log(f"  Preserved nodes: {self.preserved_num}")
        log(f"  Threads: {self.num_thread}")
        log("=" * 60)

    def check_route(self) -> None:
        """Check routing validity and report issues."""
        issues = 0
        # 1) Overuse check
        overused = []
        for node in self.routing_graph.nodes:
            if node.occupancy > node.capacity:
                overused.append((node.id, node.occupancy, node.capacity))
        if overused:
            issues += 1
            log(f"[CHECK] Overused nodes: {len(overused)} (showing up to 10)")
            for nid, occ, cap in overused[:10]:
                log(f"       Node {nid} occupancy={occ} cap={cap}")

        # 2) Path continuity per connection
        broken = 0
        for net in self.nets:
            for conn in net.connections:
                if not conn.is_routed:
                    continue
                rnodes = conn.route_nodes
                if len(rnodes) < 2:
                    broken += 1
                    continue
                ok = True
                for i in range(len(rnodes) - 1):
                    cur = rnodes[i]
                    nxt = rnodes[i + 1]
                    if nxt not in cur.children:
                        ok = False
                        break
                if not ok:
                    broken += 1
        if broken:
            issues += 1
            log(f"[CHECK] Broken connections (non-adjacent pairs): {broken}")

        if issues == 0:
            log("[CHECK] Routing validity passed: no issues found")
        else:
            log(f"[CHECK] Routing validity found {issues} issue category(ies)")
