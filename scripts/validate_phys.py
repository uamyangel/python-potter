#!/usr/bin/env python3
"""
Validate a routed Physical Netlist (.phys) produced by Python Potter.

Checks:
- File can be parsed via pycapnp
- Counts nets and stubs; prints summary of nets with non-empty PIP chains
"""

import sys
from pathlib import Path

try:
    from src.schemas import PhysicalNetlist as PN
except Exception as e:
    print("Failed to import schemas/pycapnp:", e)
    sys.exit(1)

import gzip


def load_phys(path: Path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
    return PN.PhysNetlist.from_bytes(data, traversal_limit_in_words=2**63 - 1)


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_phys.py <phys_file>")
        return 2
    out_path = Path(sys.argv[1])
    if not out_path.exists():
        print("File not found:", out_path)
        return 2

    reader = load_phys(out_path)
    nets = reader.physNets
    total = len(nets)
    with_stubs = 0
    non_empty_pips = 0
    for n in nets:
        if len(n.stubs) > 0:
            with_stubs += 1
            # Inspect first stub for a pip chain
            # Walk branches linearly
            b = n.stubs[0]
            steps = 0
            while True:
                w = b.routeSegment.which()
                if w == 'pip':
                    steps += 1
                if len(b.branches) == 0:
                    break
                b = b.branches[0]
            if steps > 0:
                non_empty_pips += 1

    print(f"Nets: {total}")
    print(f"Nets with stubs: {with_stubs}")
    print(f"Nets with non-empty pip chains (first stub): {non_empty_pips}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

