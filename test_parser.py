#!/usr/bin/env python3
"""Test script for device and netlist parsing."""

import sys
from pathlib import Path
from src.db.device import Device
from src.db.device_parser import DeviceParser
from src.db.route_node_graph import RouteNodeGraph
from src.utils.log import log

def test_device_parsing():
    """Test device file parsing."""
    log("=" * 80)
    log("Testing Device Parsing")
    log("=" * 80)

    device_file = "device/xcvu3p.device"
    device_path = Path(device_file)

    if not device_path.exists():
        log(f"Device file not found: {device_file}")
        log("Please ensure device/xcvu3p.device exists")
        return False

    try:
        # Create routing graph and device
        routing_graph = RouteNodeGraph()
        device = Device(routing_graph)

        # Read device file
        log(f"\nReading device file: {device_file}")
        device.read(device_file)

        # Print statistics
        log(f"\nDevice Statistics:")
        log(f"  Name: {device.name}")
        log(f"  Tiles: {device.num_tiles:,}")
        log(f"  Nodes: {device.routing_graph.num_nodes:,}")
        log(f"  Edges: {device.routing_graph.num_edges:,}")

        # Check some node attributes
        if device.routing_graph.num_nodes > 0:
            sample_nodes = device.routing_graph.nodes[:min(10, device.routing_graph.num_nodes)]
            log(f"\nSample Nodes (first 10):")
            for i, node in enumerate(sample_nodes):
                log(f"  Node {i}:")
                log(f"    ID: {node.id}")
                log(f"    Position: ({node.tile_x}, {node.tile_y})")
                log(f"    Type: {node.node_type}")
                log(f"    Intent: {node.intent_code}")
                log(f"    Base Cost: {node.base_cost:.2f}")
                log(f"    Length: {node.length}")

        log("\nDevice parsing test PASSED")
        return True

    except Exception as e:
        log(f"\nDevice parsing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_netlist_parsing():
    """Test netlist file parsing."""
    log("\n" + "=" * 80)
    log("Testing Netlist Parsing")
    log("=" * 80)

    # First need device loaded
    device_file = "device/xcvu3p.device"
    netlist_file = "benchmarks/logicnets_jscl_unrouted.phys"

    device_path = Path(device_file)
    netlist_path = Path(netlist_file)

    if not device_path.exists():
        log(f"Device file not found: {device_file}")
        return False

    if not netlist_path.exists():
        log(f"Netlist file not found: {netlist_file}")
        log("Available benchmarks:")
        bench_dir = Path("benchmarks")
        if bench_dir.exists():
            for f in bench_dir.glob("*.phys"):
                log(f"  {f.name}")
        return False

    try:
        # Load device first
        log(f"\nLoading device: {device_file}")
        routing_graph = RouteNodeGraph()
        device = Device(routing_graph)
        device.read(device_file)
        log(f"Device loaded: {device.routing_graph.num_nodes:,} nodes")

        # Load netlist
        log(f"\nLoading netlist: {netlist_file}")
        from src.db.netlist import Netlist
        from src.global_defs import Box

        layout = Box(0, 0, 108, 300)
        netlist = Netlist(
            device=device,
            nets=[],
            indirect_connections=[],
            direct_connections=[],
            preserved_nodes=[False] * device.routing_graph.num_nodes,
            routing_graph=device.routing_graph,
            layout=layout
        )

        netlist.read(str(netlist_path))

        # Print statistics
        log(f"\nNetlist Statistics:")
        log(f"  Nets: {len(netlist.nets):,}")
        log(f"  Indirect Connections: {len(netlist.indirect_connections):,}")
        log(f"  Direct Connections: {len(netlist.direct_connections):,}")
        log(f"  Total Connections: {len(netlist.indirect_connections) + len(netlist.direct_connections):,}")

        # Sample some connections
        if len(netlist.indirect_connections) > 0:
            log(f"\nSample Connections (first 5):")
            for i, conn in enumerate(netlist.indirect_connections[:5]):
                log(f"  Connection {i}:")
                log(f"    Net ID: {conn.net_id}")
                log(f"    Source: Node {conn.source_node.id} at ({conn.source_node.tile_x}, {conn.source_node.tile_y})")
                log(f"    Sink: Node {conn.sink_node.id} at ({conn.sink_node.tile_x}, {conn.sink_node.tile_y})")
                log(f"    BBox: ({conn.x_min}, {conn.y_min}) to ({conn.x_max}, {conn.y_max})")
                log(f"    HPWL: {conn.hpwl:.2f}")

        log("\nNetlist parsing test PASSED")
        return True

    except Exception as e:
        log(f"\nNetlist parsing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    log("Potter Python Implementation - Parser Test Suite")
    log("=" * 80)

    results = []

    # Test device parsing
    results.append(("Device Parsing", test_device_parsing()))

    # Test netlist parsing (commented out until device parsing works)
    # results.append(("Netlist Parsing", test_netlist_parsing()))

    # Print summary
    log("\n" + "=" * 80)
    log("Test Summary")
    log("=" * 80)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        log(f"  {name}: {status}")

    all_passed = all(passed for _, passed in results)
    log(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
