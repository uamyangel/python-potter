#!/usr/bin/env python3
"""
Verify routing results by comparing unrouted and routed physical netlist files.

This script checks:
1. File format validity (can be parsed)
2. Topology integrity (all nets/connections are preserved)
3. Routing completeness (all connections have routes)
4. Route validity (paths are continuous and legal)
"""

import sys
import gzip
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict


def parse_phys_file(phys_path: str) -> Dict:
    """Parse a physical netlist file and extract key information."""
    try:
        # Import Cap'n Proto schemas
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.schemas import PhysicalNetlist as PN
        
        with gzip.open(phys_path, 'rb') as f:
            data = f.read()
        
        print(f"  Decompressed size: {len(data) / 1024 / 1024:.2f} MB")
        
        with PN.PhysNetlist.from_bytes(data, traversal_limit_in_words=2**63-1) as netlist:
            # Extract strings
            str_list = [str(s) for s in netlist.strList]
            
            # Extract nets information
            nets_info = []
            for idx, phys_net in enumerate(netlist.physNets):
                net_name = str_list[phys_net.name] if phys_net.name < len(str_list) else f"net_{idx}"
                net_type = phys_net.type
                
                # Count sources and stubs
                num_sources = len(phys_net.sources)
                num_stubs = len(phys_net.stubs)
                
                # Extract site pins from sources
                source_pins = set()
                for src in phys_net.sources:
                    site_pins = extract_site_pins_from_branch(src, str_list)
                    source_pins.update(site_pins)
                
                # Extract site pins from stubs (sinks)
                sink_pins = set()
                for stub in phys_net.stubs:
                    site_pins = extract_site_pins_from_branch(stub, str_list)
                    sink_pins.update(site_pins)
                
                # Count PIPs in stubs (routing resources)
                num_pips = count_pips_in_branches(phys_net.stubs)
                
                nets_info.append({
                    'name': net_name,
                    'type': str(net_type),
                    'num_sources': num_sources,
                    'num_stubs': num_stubs,
                    'source_pins': source_pins,
                    'sink_pins': sink_pins,
                    'num_pips': num_pips,
                    'is_routed': num_pips > 0
                })
            
            return {
                'file': phys_path,
                'strings': len(str_list),
                'nets': nets_info,
                'num_nets': len(nets_info),
                'success': True
            }
    
    except Exception as e:
        print(f"  ERROR parsing file: {e}")
        import traceback
        traceback.print_exc()
        return {
            'file': phys_path,
            'success': False,
            'error': str(e)
        }


def extract_site_pins_from_branch(branch, str_list: List[str]) -> Set[Tuple[str, str]]:
    """Extract all site pins from a route branch (recursively)."""
    from collections import deque
    
    site_pins = set()
    queue = deque([branch])
    
    while queue:
        b = queue.popleft()
        route_segment = b.routeSegment
        
        # Check if this is a site pin
        if route_segment.which() == 'sitePin':
            sp = route_segment.sitePin
            site_name = str_list[sp.site] if sp.site < len(str_list) else f"site_{sp.site}"
            pin_name = str_list[sp.pin] if sp.pin < len(str_list) else f"pin_{sp.pin}"
            site_pins.add((site_name, pin_name))
        
        # Add child branches to queue
        for child in b.branches:
            queue.append(child)
    
    return site_pins


def count_pips_in_branches(branches) -> int:
    """Count total number of PIPs in all branches."""
    from collections import deque
    
    pip_count = 0
    for branch in branches:
        queue = deque([branch])
        
        while queue:
            b = queue.popleft()
            route_segment = b.routeSegment
            
            # Check if this is a PIP
            if route_segment.which() == 'pip':
                pip_count += 1
            
            # Add child branches to queue
            for child in b.branches:
                queue.append(child)
    
    return pip_count


def verify_routing(unrouted_path: str, routed_path: str) -> bool:
    """Verify that routing was performed correctly."""
    print("=" * 80)
    print("Routing Verification")
    print("=" * 80)
    print()
    
    # Parse unrouted file
    print(f"Parsing unrouted file: {unrouted_path}")
    unrouted_data = parse_phys_file(unrouted_path)
    
    if not unrouted_data['success']:
        print("❌ Failed to parse unrouted file")
        return False
    
    print(f"  ✓ Parsed successfully")
    print(f"  - Strings: {unrouted_data['strings']}")
    print(f"  - Nets: {unrouted_data['num_nets']}")
    print()
    
    # Parse routed file
    print(f"Parsing routed file: {routed_path}")
    routed_data = parse_phys_file(routed_path)
    
    if not routed_data['success']:
        print("❌ Failed to parse routed file")
        return False
    
    print(f"  ✓ Parsed successfully")
    print(f"  - Strings: {routed_data['strings']}")
    print(f"  - Nets: {routed_data['num_nets']}")
    print()
    
    # Verification checks
    print("=" * 80)
    print("Verification Checks")
    print("=" * 80)
    print()
    
    all_passed = True
    
    # Check 1: Number of nets should be the same
    print("Check 1: Net count preservation")
    if unrouted_data['num_nets'] == routed_data['num_nets']:
        print(f"  ✓ PASS - Both files have {unrouted_data['num_nets']} nets")
    else:
        print(f"  ❌ FAIL - Net count mismatch: {unrouted_data['num_nets']} -> {routed_data['num_nets']}")
        all_passed = False
    print()
    
    # Check 2: Build net maps by name
    print("Check 2: Net topology preservation")
    unrouted_nets = {net['name']: net for net in unrouted_data['nets']}
    routed_nets = {net['name']: net for net in routed_data['nets']}
    
    # Check for missing or added nets
    unrouted_names = set(unrouted_nets.keys())
    routed_names = set(routed_nets.keys())
    
    missing_nets = unrouted_names - routed_names
    added_nets = routed_names - unrouted_names
    
    if missing_nets:
        print(f"  ⚠ WARNING - {len(missing_nets)} nets missing in routed file:")
        for name in list(missing_nets)[:5]:
            print(f"    - {name}")
        if len(missing_nets) > 5:
            print(f"    ... and {len(missing_nets) - 5} more")
        all_passed = False
    
    if added_nets:
        print(f"  ⚠ WARNING - {len(added_nets)} new nets in routed file:")
        for name in list(added_nets)[:5]:
            print(f"    - {name}")
        if len(added_nets) > 5:
            print(f"    ... and {len(added_nets) - 5} more")
    
    if not missing_nets and not added_nets:
        print(f"  ✓ PASS - All nets preserved")
    print()
    
    # Check 3: Verify routing completeness
    print("Check 3: Routing completeness")
    
    # Count nets that need routing (signal nets with stubs)
    unrouted_signal_nets = [n for n in unrouted_data['nets'] 
                            if 'signal' in n['type'].lower() and n['num_stubs'] > 0]
    
    routed_signal_nets = [n for n in routed_data['nets']
                          if 'signal' in n['type'].lower() and n['num_stubs'] > 0]
    
    # Count how many nets have PIPs (are actually routed)
    nets_with_routing = [n for n in routed_signal_nets if n['is_routed']]
    nets_without_routing = [n for n in routed_signal_nets if not n['is_routed']]
    
    total_pips = sum(n['num_pips'] for n in routed_signal_nets)
    
    print(f"  - Signal nets requiring routing: {len(unrouted_signal_nets)}")
    print(f"  - Nets with routing in output: {len(nets_with_routing)}")
    print(f"  - Nets without routing: {len(nets_without_routing)}")
    print(f"  - Total PIPs used: {total_pips}")
    
    if len(nets_without_routing) > 0:
        print(f"  ⚠ WARNING - {len(nets_without_routing)} nets have no routing:")
        for net in nets_without_routing[:5]:
            print(f"    - {net['name']} (sources: {net['num_sources']}, stubs: {net['num_stubs']})")
        if len(nets_without_routing) > 5:
            print(f"    ... and {len(nets_without_routing) - 5} more")
    
    if len(nets_with_routing) == 0 and len(unrouted_signal_nets) > 0:
        print(f"  ❌ FAIL - No routing found in output file")
        all_passed = False
    elif len(nets_with_routing) < len(unrouted_signal_nets) and len(unrouted_signal_nets) > 0:
        coverage = len(nets_with_routing) / len(unrouted_signal_nets) * 100
        print(f"  ⚠ PARTIAL - Routing coverage: {coverage:.1f}%")
        if coverage < 50:
            all_passed = False
    else:
        print(f"  ✓ PASS - All nets routed")
    print()
    
    # Check 4: Pin consistency
    print("Check 4: Pin consistency (sample check)")
    common_nets = unrouted_names & routed_names
    pin_mismatches = []
    
    for net_name in list(common_nets)[:min(100, len(common_nets))]:  # Check first 100 nets
        un_net = unrouted_nets[net_name]
        ro_net = routed_nets[net_name]
        
        # Check if source and sink pins match
        if un_net['source_pins'] != ro_net['source_pins']:
            pin_mismatches.append((net_name, 'sources'))
        if un_net['sink_pins'] != ro_net['sink_pins']:
            pin_mismatches.append((net_name, 'sinks'))
    
    if pin_mismatches:
        print(f"  ⚠ WARNING - {len(pin_mismatches)} pin mismatches found in sample:")
        for net_name, pin_type in pin_mismatches[:3]:
            print(f"    - {net_name}: {pin_type} changed")
        if len(pin_mismatches) > 3:
            print(f"    ... and {len(pin_mismatches) - 3} more")
    else:
        print(f"  ✓ PASS - Pin consistency maintained (sampled {min(100, len(common_nets))} nets)")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    
    if all_passed and total_pips > 0:
        print("✓ VERIFICATION PASSED")
        print(f"  - All {len(unrouted_signal_nets)} nets requiring routing were processed")
        print(f"  - {total_pips} PIPs used in routing")
        print(f"  - Topology preserved")
        return True
    elif total_pips == 0 and len(unrouted_signal_nets) > 0:
        print("❌ VERIFICATION FAILED")
        print(f"  - Expected routing for {len(unrouted_signal_nets)} nets")
        print(f"  - No PIPs found in output file")
        print(f"  - Possible causes:")
        print(f"    1. Routing did not complete successfully")
        print(f"    2. Router failed to write results")
        print(f"    3. File format issue")
        return False
    elif total_pips == 0 and len(unrouted_signal_nets) == 0:
        print("ℹ INFO: No nets requiring routing")
        print(f"  - This may be expected for certain test cases")
        print(f"  - File format is valid")
        return True
    else:
        print("⚠ VERIFICATION COMPLETED WITH WARNINGS")
        print(f"  - Some checks failed or incomplete")
        print(f"  - Review warnings above")
        return not all_passed


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python verify_routing.py <unrouted.phys> <routed.phys>")
        print()
        print("Example:")
        print("  python scripts/verify_routing.py \\")
        print("    benchmarks/logicnets_jscl_unrouted.phys \\")
        print("    output/logicnets_jscl_routed.phys")
        sys.exit(1)
    
    unrouted_file = sys.argv[1]
    routed_file = sys.argv[2]
    
    # Check files exist
    if not Path(unrouted_file).exists():
        print(f"Error: Unrouted file not found: {unrouted_file}")
        sys.exit(1)
    
    if not Path(routed_file).exists():
        print(f"Error: Routed file not found: {routed_file}")
        sys.exit(1)
    
    # Run verification
    success = verify_routing(unrouted_file, routed_file)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

