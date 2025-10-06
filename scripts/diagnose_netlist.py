#!/usr/bin/env python3
"""
Diagnose netlist parsing issues.
Check why nets are not being extracted properly.
"""

import sys
import gzip
from pathlib import Path


def diagnose_netlist(netlist_path: str, device_path: str = None):
    """Diagnose netlist file parsing."""
    print("=" * 80)
    print("Netlist Parsing Diagnostic")
    print("=" * 80)
    print()
    
    # Parse netlist
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.schemas import PhysicalNetlist as PN
    
    print(f"Parsing: {netlist_path}")
    with gzip.open(netlist_path, 'rb') as f:
        data = f.read()
    
    with PN.PhysNetlist.from_bytes(data, traversal_limit_in_words=2**63-1) as netlist:
        str_list = [str(s) for s in netlist.strList]
        print(f"Strings: {len(str_list)}")
        print(f"Physical Nets: {len(netlist.physNets)}")
        print()
        
        # Analyze nets
        signal_nets = []
        for idx, phys_net in enumerate(netlist.physNets):
            net_name = str_list[phys_net.name] if phys_net.name < len(str_list) else f"net_{idx}"
            net_type = phys_net.type
            
            if net_type == PN.PhysNetlist.NetType.signal:
                num_sources = len(phys_net.sources)
                num_stubs = len(phys_net.stubs)
                
                # Extract site pins
                source_pins = []
                for src in phys_net.sources:
                    pins = extract_site_pins(src, str_list)
                    source_pins.extend(pins)
                
                sink_pins = []
                for stub in phys_net.stubs:
                    pins = extract_site_pins(stub, str_list)
                    sink_pins.extend(pins)
                
                signal_nets.append({
                    'name': net_name,
                    'sources': num_sources,
                    'stubs': num_stubs,
                    'source_pins': source_pins,
                    'sink_pins': sink_pins
                })
        
        print(f"Signal Nets: {len(signal_nets)}")
        print()
        
        # Show first few signal nets
        print("Sample Signal Nets (first 5):")
        print("-" * 80)
        for i, net in enumerate(signal_nets[:5]):
            print(f"\nNet {i+1}: {net['name']}")
            print(f"  Sources: {net['sources']}, Stubs: {net['stubs']}")
            if net['source_pins']:
                print(f"  Source pins ({len(net['source_pins'])}):")
                for site, pin in net['source_pins'][:3]:
                    print(f"    - {site} / {pin}")
                if len(net['source_pins']) > 3:
                    print(f"    ... and {len(net['source_pins']) - 3} more")
            if net['sink_pins']:
                print(f"  Sink pins ({len(net['sink_pins'])}):")
                for site, pin in net['sink_pins'][:3]:
                    print(f"    - {site} / {pin}")
                if len(net['sink_pins']) > 3:
                    print(f"    ... and {len(net['sink_pins']) - 3} more")
        
        print()
        print("=" * 80)
        
        # If device is provided, check site resolution
        if device_path and Path(device_path).exists():
            print("\nChecking Device Site Resolution")
            print("=" * 80)
            print()
            
            from src.db.device import Device
            from src.db.route_node_graph import RouteNodeGraph
            
            print(f"Loading device: {device_path}")
            # Create routing graph first, then pass to Device
            routing_graph = RouteNodeGraph()
            device = Device(routing_graph)
            device.read(device_path)
            print(f"Device loaded: {len(device.sites)} sites, {device.routing_graph.num_nodes} nodes")
            print()
            
            # Check if we can resolve site pins
            print("Checking site pin resolution for sample nets...")
            print("-" * 80)
            
            found_count = 0
            not_found_count = 0
            
            for i, net in enumerate(signal_nets[:5]):
                print(f"\nNet: {net['name']}")
                
                # Check source pins
                for site, pin in net['source_pins'][:2]:
                    node_idx = device.get_site_pin_node(site, pin)
                    if node_idx is not None:
                        print(f"  ✓ Found source: {site}/{pin} -> node {node_idx}")
                        found_count += 1
                    else:
                        print(f"  ✗ NOT FOUND source: {site}/{pin}")
                        not_found_count += 1
                        
                        # Try to diagnose
                        if site not in device.site_name_to_idx:
                            print(f"    Reason: Site '{site}' not in device database")
                        else:
                            site_idx = device.site_name_to_idx[site]
                            site_info = device.sites[site_idx]
                            site_type_idx = site_info['site_type_idx']
                            if site_type_idx < len(device.site_type_pin_name_to_idx):
                                available_pins = device.site_type_pin_name_to_idx[site_type_idx].keys()
                                print(f"    Reason: Pin '{pin}' not found in site type")
                                print(f"    Available pins: {list(available_pins)[:10]}")
                            else:
                                print(f"    Reason: Invalid site type index {site_type_idx}")
                
                # Check sink pins
                for site, pin in net['sink_pins'][:2]:
                    node_idx = device.get_site_pin_node(site, pin)
                    if node_idx is not None:
                        print(f"  ✓ Found sink: {site}/{pin} -> node {node_idx}")
                        found_count += 1
                    else:
                        print(f"  ✗ NOT FOUND sink: {site}/{pin}")
                        not_found_count += 1
            
            print()
            print("-" * 80)
            print(f"Summary: {found_count} pins found, {not_found_count} NOT FOUND")
            print()
            
            if not_found_count > 0:
                print("⚠ ISSUE DETECTED: Some site pins cannot be resolved")
                print("  This explains why nets are not being extracted.")
                print("  Possible causes:")
                print("  1. Netlist and device file mismatch (different parts/versions)")
                print("  2. Site naming convention differences")
                print("  3. Incomplete device database")
            else:
                print("✓ All sampled pins resolved successfully")
                print("  The issue may be in later parsing stages")


def extract_site_pins(branch, str_list):
    """Extract site pins from a branch."""
    from collections import deque
    
    site_pins = []
    queue = deque([branch])
    
    while queue:
        b = queue.popleft()
        route_segment = b.routeSegment
        
        if route_segment.which() == 'sitePin':
            sp = route_segment.sitePin
            site_name = str_list[sp.site] if sp.site < len(str_list) else f"site_{sp.site}"
            pin_name = str_list[sp.pin] if sp.pin < len(str_list) else f"pin_{sp.pin}"
            site_pins.append((site_name, pin_name))
        
        for child in b.branches:
            queue.append(child)
    
    return site_pins


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python diagnose_netlist.py <netlist.phys> [device.device]")
        print()
        print("Example:")
        print("  python scripts/diagnose_netlist.py \\")
        print("    benchmarks/logicnets_jscl_unrouted.phys \\")
        print("    device/xcvu3p.device")
        sys.exit(1)
    
    netlist_file = sys.argv[1]
    device_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(netlist_file).exists():
        print(f"Error: Netlist file not found: {netlist_file}")
        sys.exit(1)
    
    diagnose_netlist(netlist_file, device_file)


if __name__ == '__main__':
    main()

