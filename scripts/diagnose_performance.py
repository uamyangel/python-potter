#!/usr/bin/env python3
"""Performance diagnosis script for Python Potter."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.database import Database
from src.utils.log import log
import time


def diagnose_database(device_file: str, netlist_file: str):
    """Diagnose database loading and basic statistics."""
    log("=" * 80)
    log("Potter Performance Diagnostic")
    log("=" * 80)

    database = Database()
    database.set_num_thread(4)

    # Load device
    log("\n[1] Loading device...")
    t0 = time.time()
    database.read_device(device_file)
    device_load_time = time.time() - t0
    log(f"Device load time: {device_load_time:.2f}s")

    # Check routing graph
    graph = database.routing_graph
    log(f"\n[2] Routing Graph Statistics:")
    log(f"  Nodes: {graph.num_nodes:,}")
    log(f"  Edges: {graph.num_edges:,}")
    log(f"  Has CSR: {getattr(graph, '_has_csr', False)}")

    if graph.num_nodes == 0:
        log("  ⚠️  WARNING: Routing graph is EMPTY! Device not loaded properly.")
        return

    # Sample a few nodes
    if graph.nodes:
        log(f"\n[3] Sample Nodes (first 5):")
        for i, node in enumerate(graph.nodes[:5]):
            log(f"  Node {node.id}: tile=({node.tile_x},{node.tile_y}), "
                f"children={len(node.children)}, type={node.node_type}")

    # Average degree
    if graph.nodes:
        avg_degree = sum(len(n.children) for n in graph.nodes) / len(graph.nodes)
        log(f"\n[4] Average node degree: {avg_degree:.2f}")

    # Load netlist
    log(f"\n[5] Loading netlist...")
    t0 = time.time()
    database.read_netlist(netlist_file)
    netlist_load_time = time.time() - t0
    log(f"Netlist load time: {netlist_load_time:.2f}s")

    log(f"\n[6] Netlist Statistics:")
    log(f"  Nets: {len(database.nets):,}")
    log(f"  Indirect connections: {len(database.indirect_connections):,}")
    log(f"  Direct connections: {len(database.direct_connections):,}")

    if len(database.indirect_connections) == 0:
        log("  ⚠️  WARNING: No indirect connections! Netlist may be empty or parser failed.")
        return

    # Sample connections
    if database.indirect_connections:
        log(f"\n[7] Sample Connections (first 5):")
        for i, conn in enumerate(database.indirect_connections[:5]):
            log(f"  Conn {conn.id}: net={conn.net_id}, "
                f"bbox=({conn.x_min},{conn.y_min})-({conn.x_max},{conn.y_max}), "
                f"hpwl={conn.hpwl:.1f}")

    # Estimate routing complexity
    if database.indirect_connections:
        total_hpwl = sum(c.hpwl for c in database.indirect_connections)
        avg_hpwl = total_hpwl / len(database.indirect_connections)
        log(f"\n[8] Routing Complexity:")
        log(f"  Total HPWL: {total_hpwl:,.0f}")
        log(f"  Average HPWL: {avg_hpwl:.2f}")

    # Estimate iteration time
    num_conns = len(database.indirect_connections)
    avg_fanout = 3  # Typical assumption
    nodes_per_conn = avg_hpwl * 50  # Very rough estimate
    total_node_visits = num_conns * nodes_per_conn

    log(f"\n[9] Estimated Work per Iteration:")
    log(f"  Connections to route: {num_conns:,}")
    log(f"  Est. node visits per iteration: {total_node_visits:,.0f}")

    # Estimate Python overhead
    python_overhead_ns = 100  # ns per attribute access
    total_overhead_s = (total_node_visits * python_overhead_ns) / 1e9
    log(f"  Est. Python overhead per iteration: {total_overhead_s:.1f}s (at 100ns/access)")

    if total_overhead_s > 60:
        log(f"  ⚠️  WARNING: Single iteration estimated at {total_overhead_s/60:.1f} minutes!")
        log(f"      This is too slow. Need optimization!")

    log("\n[10] Recommendations:")
    if not getattr(graph, '_has_csr', False):
        log("  - Build CSR adjacency for faster neighbor iteration")
    if total_overhead_s > 10:
        log("  - Use NumPy arrays for node attributes")
        log("  - Remove Python locks from hot path")
        log("  - Consider Cython compilation for A* core")

    log("\n" + "=" * 80)
    log("Diagnosis complete.")
    log("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Diagnose Potter performance")
    parser.add_argument("-d", "--device", default="device/xcvu3p.device",
                        help="Device file path")
    parser.add_argument("-i", "--input", required=True,
                        help="Input netlist file")
    args = parser.parse_args()

    try:
        diagnose_database(args.device, args.input)
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
