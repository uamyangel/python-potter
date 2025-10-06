#!/usr/bin/env python3
"""
Quick test to verify prepare_database_state() is pickle-safe.

This tests the fix for the RecursionError during multiprocessing.
"""

import sys
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.db.database import Database
from src.route.process_worker import prepare_database_state


def test_pickle_state():
    """Test that database state can be pickled without RecursionError."""

    print("Creating minimal database for testing...")
    database = Database()

    # Set minimal required attributes
    database.routing_graph.num_nodes = 10
    database.routing_graph.num_edges = 20

    # Build CSR (required for multiprocessing)
    try:
        database.routing_graph.build_csr()
    except Exception as e:
        print(f"Warning: Could not build CSR (expected for minimal test): {e}")
        # Create minimal CSR manually
        database.routing_graph.csr_indptr = [0] * 11  # num_nodes + 1
        database.routing_graph.csr_indices = []
        database.routing_graph._has_csr = True

    # Build NumPy arrays (required for multiprocessing)
    try:
        database.routing_graph.build_numpy_arrays()
    except Exception as e:
        print(f"Warning: Could not build NumPy arrays (expected for minimal test): {e}")
        # Create minimal NumPy arrays manually
        import numpy as np
        database.routing_graph.tile_x_arr = np.zeros(10, dtype=np.int32)
        database.routing_graph.tile_y_arr = np.zeros(10, dtype=np.int32)
        database.routing_graph.base_cost_arr = np.ones(10, dtype=np.float32)
        database.routing_graph.length_arr = np.ones(10, dtype=np.int16)
        database.routing_graph.node_type_arr = np.zeros(10, dtype=np.int16)
        database.routing_graph.intent_code_arr = np.zeros(10, dtype=np.int16)
        database.routing_graph.capacity_arr = np.ones(10, dtype=np.int8)
        database.routing_graph.pres_cong_cost_arr = np.ones(10, dtype=np.float32)
        database.routing_graph.hist_cong_cost_arr = np.ones(10, dtype=np.float32)
        database.routing_graph.is_accessible_arr = np.ones(10, dtype=np.bool_)
        database.routing_graph._numpy_arrays_built = True

    print("\nPreparing database state...")
    try:
        state = prepare_database_state(database)
        print(f"✓ Database state prepared successfully")
        print(f"  Keys: {list(state.keys())}")
        print(f"  num_nodes: {state.get('num_nodes', 'N/A')}")
        print(f"  num_edges: {state.get('num_edges', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed to prepare database state: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nTesting pickle serialization...")
    try:
        serialized = pickle.dumps(state)
        print(f"✓ Pickle serialization succeeded ({len(serialized)} bytes)")
    except RecursionError as e:
        print(f"✗ FAILED: RecursionError during pickle!")
        print(f"  This means circular references still exist")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nTesting pickle deserialization...")
    try:
        deserialized = pickle.loads(serialized)
        print(f"✓ Pickle deserialization succeeded")
        print(f"  num_nodes matches: {deserialized['num_nodes'] == state['num_nodes']}")
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED - Database state is pickle-safe!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_pickle_state()
    sys.exit(0 if success else 1)
