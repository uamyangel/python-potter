#!/usr/bin/env python3
"""Potter: A parallel overlap-tolerant FPGA router for Xilinx UltraScale FPGAs.

This is the main entry point for the Potter router.
"""

import argparse
import sys
from pathlib import Path

from src.db.database import Database
from src.route.router import Router
from src.utils.log import log


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Potter - An Open-Source High-Concurrency and High-Performance "
                   "Parallel Router for UltraScale FPGAs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="The input (unrouted) physical netlist (.phys file)"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="The output (routed) physical netlist (.phys file)"
    )

    parser.add_argument(
        "-d", "--device",
        default="device/xcvu3p.device",
        help="The device file (default: device/xcvu3p.device)"
    )

    parser.add_argument(
        "-t", "--thread",
        type=int,
        default=32,
        help="The number of threads (default: 32)"
    )

    parser.add_argument(
        "-r", "--runtime_first",
        action="store_true",
        help="Enable runtime first mode (default: stability first mode)"
    )

    parser.add_argument(
        "--use_threads",
        action="store_true",
        help="Use threading instead of multiprocessing (default: multiprocessing for better performance)"
    )

    parser.add_argument(
        "--device_cache",
        choices=["off", "light"],
        default="light",
        help="Device cache mode: 'light' caches indices/arrays for faster reloads"
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=500,
        help="Maximum routing iterations (default: 500)"
    )

    parser.add_argument(
        "--pres_fac_init",
        type=float,
        default=0.5,
        help="Initial present congestion factor (default: 0.5)"
    )

    parser.add_argument(
        "--pres_fac_mult",
        type=float,
        default=1.8,
        help="Multiplicative increase of present congestion factor per iteration (default: 1.8)"
    )

    parser.add_argument(
        "--hist_fac",
        type=float,
        default=1.0,
        help="Historical congestion factor increment per overuse (default: 1.0)"
    )

    parser.add_argument(
        "--x_margin",
        type=int,
        default=3,
        help="Fixed X margin for connection routing window (default: 3)"
    )

    parser.add_argument(
        "--y_margin",
        type=int,
        default=15,
        help="Fixed Y margin for connection routing window (default: 15)"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Print banner
    log("=" * 60)
    log("Potter - Parallel Overlap-Tolerant Router")
    log("Python Implementation (Optimized)")
    log("=" * 60)
    log(f"Input:  {args.input}")
    log(f"Output: {args.output}")
    log(f"Device: {args.device}")
    log(f"Threads: {args.thread}")
    log(f"Mode: {'Runtime-first' if args.runtime_first else 'Stability-first'}")
    log(f"Parallelism: {'Threading' if args.use_threads else 'Multiprocessing (GIL-free)'}")
    log()

    # Validate input files
    if not Path(args.input).exists():
        log(f"ERROR: Input file not found: {args.input}")
        return 1

    if not Path(args.device).exists():
        log(f"ERROR: Device file not found: {args.device}")
        return 1

    try:
        # Initialize database
        database = Database()
        database.set_num_thread(args.thread)

        # Load device and netlist
        database.set_device_cache_mode(args.device_cache)
        database.read_device(args.device)
        database.read_netlist(args.input)
        database.set_route_node_children()
        database.print_statistic()

        # Route
        router = Router(database, args.runtime_first, use_multiprocessing=not args.use_threads)
        # Allow overriding iteration count from CLI
        if hasattr(args, 'max_iter') and args.max_iter is not None:
            router.max_iter = max(1, int(args.max_iter))
        # Override congestion schedule from CLI
        router.present_cong_factor = float(args.pres_fac_init)
        router.present_cong_multiplier = float(args.pres_fac_mult)
        router.historical_cong_factor = float(args.hist_fac)
        # Precompute routing windows with given margins
        router._prepare_connection_bboxes(x_margin=int(args.x_margin), y_margin=int(args.y_margin))
        router.route()

        # Optional route validity check
        database.check_route()

        # Write output
        database.write_netlist(args.output, router.node_routing_results)

        # Check memory
        database.device.check_memory_peak(-1)

        log()
        log("Routing completed successfully!")
        return 0

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
