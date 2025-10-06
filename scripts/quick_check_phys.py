#!/usr/bin/env python3
"""
Quick check for physical netlist files.
Provides a fast overview of file contents without deep validation.
"""

import sys
import gzip
from pathlib import Path


def quick_check(phys_path: str):
    """Quickly check a physical netlist file and print summary."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.schemas import PhysicalNetlist as PN
        
        print(f"Checking: {phys_path}")
        print("-" * 60)
        
        # Check file exists
        if not Path(phys_path).exists():
            print("❌ File not found")
            return False
        
        file_size = Path(phys_path).stat().st_size / 1024 / 1024
        print(f"File size: {file_size:.2f} MB")
        
        # Decompress
        with gzip.open(phys_path, 'rb') as f:
            data = f.read()
        
        decompressed_size = len(data) / 1024 / 1024
        print(f"Decompressed: {decompressed_size:.2f} MB")
        
        # Parse
        with PN.PhysNetlist.from_bytes(data, traversal_limit_in_words=2**63-1) as netlist:
            str_list = [str(s) for s in netlist.strList]
            print(f"Strings: {len(str_list)}")
            
            # Count nets by type
            nets_by_type = {}
            total_sources = 0
            total_stubs = 0
            total_pips = 0
            
            for phys_net in netlist.physNets:
                net_type = str(phys_net.type)
                nets_by_type[net_type] = nets_by_type.get(net_type, 0) + 1
                
                total_sources += len(phys_net.sources)
                total_stubs += len(phys_net.stubs)
                
                # Count PIPs in stubs
                from collections import deque
                for stub in phys_net.stubs:
                    queue = deque([stub])
                    while queue:
                        b = queue.popleft()
                        if b.routeSegment.which() == 'pip':
                            total_pips += 1
                        for child in b.branches:
                            queue.append(child)
            
            print(f"Physical Nets: {len(netlist.physNets)}")
            for net_type, count in sorted(nets_by_type.items()):
                print(f"  - {net_type}: {count}")
            
            print(f"Total Sources: {total_sources}")
            print(f"Total Stubs: {total_stubs}")
            print(f"Total PIPs: {total_pips}")
            
            # Other sections
            print(f"Placements: {len(netlist.placements)}")
            print(f"Physical Cells: {len(netlist.physCells)}")
            print(f"Site Instances: {len(netlist.siteInsts)}")
            print(f"Properties: {len(netlist.properties)}")
            
            print("-" * 60)
            
            # Interpretation
            if total_pips > 0:
                print("✓ File contains routing (PIPs present)")
            else:
                print("ℹ No PIPs found - likely unrouted or test file")
            
            print("✓ File is valid and parseable")
            return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python quick_check_phys.py <file.phys> [file2.phys ...]")
        print()
        print("Example:")
        print("  python scripts/quick_check_phys.py benchmarks/*.phys")
        sys.exit(1)
    
    files = sys.argv[1:]
    results = []
    
    for phys_file in files:
        success = quick_check(phys_file)
        results.append((phys_file, success))
        print()
    
    # Summary
    if len(results) > 1:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        passed = sum(1 for _, s in results if s)
        print(f"Checked {len(results)} files: {passed} valid, {len(results) - passed} failed")
    
    sys.exit(0 if all(s for _, s in results) else 1)


if __name__ == '__main__':
    main()

