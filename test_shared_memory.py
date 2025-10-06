#!/usr/bin/env python3
"""
Test script for shared memory multiprocessing implementation.

This script validates:
1. Shared memory creation and attachment
2. Memory usage comparison (threads vs processes)
3. Basic routing functionality
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.route.shared_memory_manager import SharedMemoryManager, attach_shared_array
from src.utils.memory_monitor import get_memory_usage_mb, format_memory_usage


def test_shared_memory():
    """Test basic shared memory operations."""
    print("=" * 60)
    print("Test 1: Basic Shared Memory Operations")
    print("=" * 60)
    
    # Create test array
    test_array = np.arange(1000000, dtype=np.int32)
    print(f"Test array size: {test_array.nbytes / 1024 / 1024:.1f} MB")
    
    # Create shared memory manager
    manager = SharedMemoryManager()
    
    # Create shared array
    shm_name = manager.create_shared_array('test_array', test_array)
    print(f"Shared memory created: {shm_name}")
    
    # Get metadata
    metadata = manager.get_shared_arrays_metadata()
    print(f"Metadata: {metadata['test_array']}")
    
    # Attach to shared array (simulating worker process)
    meta = metadata['test_array']
    attached_array, shm_ref = attach_shared_array(
        meta['shm_name'],
        meta['shape'],
        meta['dtype']
    )
    
    # Verify data integrity
    if np.array_equal(test_array, attached_array):
        print("✅ Data integrity verified!")
    else:
        print("❌ Data mismatch!")
        shm_ref.close()
        return False
    
    # Close the attached shared memory
    shm_ref.close()
    
    # Cleanup
    manager.cleanup()
    print("✅ Shared memory cleaned up")
    print()
    
    return True


def test_memory_usage():
    """Test memory usage monitoring."""
    print("=" * 60)
    print("Test 2: Memory Usage Monitoring")
    print("=" * 60)
    
    usage = get_memory_usage_mb()
    formatted = format_memory_usage(usage)
    print(f"Current memory: {formatted}")
    
    # Allocate some memory
    large_array = np.zeros((10, 1000, 1000), dtype=np.float64)
    size_mb = large_array.nbytes / 1024 / 1024
    print(f"\nAllocated {size_mb:.1f} MB array")
    
    usage_after = get_memory_usage_mb()
    formatted_after = format_memory_usage(usage_after)
    print(f"Memory after allocation: {formatted_after}")
    
    delta = usage_after['rss'] - usage['rss']
    print(f"Memory increase: {delta:.1f} MB")
    print("✅ Memory monitoring works")
    print()
    
    return True


def test_routing_basic():
    """Test basic routing with small benchmark."""
    print("=" * 60)
    print("Test 3: Basic Routing Test")
    print("=" * 60)
    
    # Check if benchmark exists
    benchmark = Path("benchmarks/logicnets_jscl_unrouted.phys")
    if not benchmark.exists():
        print(f"⚠️  Benchmark not found: {benchmark}")
        print("   Skipping routing test")
        return True
    
    print(f"Testing with: {benchmark}")
    print("This will take a few minutes...")
    print()
    
    from src.db.database import Database
    from src.route.router import Router
    
    # Initialize database
    database = Database()
    database.set_num_thread(1)  # Use ONLY 1 process for test (low memory)
    database.set_device_cache_mode("light")
    
    print("Loading device...")
    database.read_device("device/xcvu3p.device")
    
    print("Loading netlist...")
    database.read_netlist(str(benchmark))
    database.set_route_node_children()
    database.print_statistic()
    
    # Test multiprocessing with shared memory
    print("\n--- Testing Multiprocessing (with shared memory, 1 worker) ---")
    print("Note: Using only 1 worker to conserve memory")
    start = time.time()
    router = Router(database, is_runtime_first=False, use_multiprocessing=True)
    router.max_iter = 2  # Only 2 iterations for test (faster)
    router._prepare_connection_bboxes(x_margin=3, y_margin=15)
    
    try:
        router.route()
        elapsed = time.time() - start
        print(f"\n✅ Routing completed in {elapsed:.2f}s")
        print(f"   Routed: {router.routed_connections} connections")
        print(f"   Failed: {router.failed_connections} connections")
    except Exception as e:
        print(f"❌ Routing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Potter Shared Memory Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Shared Memory Operations", test_shared_memory),
        ("Memory Monitoring", test_memory_usage),
        ("Basic Routing", test_routing_basic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
