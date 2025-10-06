#!/usr/bin/env python3
"""
Minimal test for shared memory (without routing).

Tests only the shared memory infrastructure without loading large designs.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.route.shared_memory_manager import SharedMemoryManager, attach_shared_array
from src.utils.memory_monitor import get_memory_usage_mb, format_memory_usage


def test_shared_memory_basic():
    """Test basic shared memory operations without routing."""
    print("=" * 60)
    print("Minimal Shared Memory Test")
    print("=" * 60)
    print()
    
    # Test 1: Small array
    print("Test 1: Small array (1 MB)")
    test_array = np.arange(250000, dtype=np.int32)
    print(f"  Array size: {test_array.nbytes / 1024 / 1024:.1f} MB")
    
    manager = SharedMemoryManager()
    shm_name = manager.create_shared_array('test_small', test_array)
    print(f"  ✅ Shared memory created: {shm_name}")
    
    metadata = manager.get_shared_arrays_metadata()
    meta = metadata['test_small']
    attached_array, shm_ref = attach_shared_array(
        meta['shm_name'],
        meta['shape'],
        meta['dtype']
    )
    
    if np.array_equal(test_array, attached_array):
        print("  ✅ Data integrity verified")
    else:
        print("  ❌ Data mismatch!")
        return False
    
    shm_ref.close()
    manager.cleanup()
    print("  ✅ Cleanup successful")
    print()
    
    # Test 2: Multiple arrays
    print("Test 2: Multiple arrays (10 × 10 MB)")
    manager2 = SharedMemoryManager()
    arrays = []
    
    for i in range(10):
        arr = np.full(2500000, i, dtype=np.int32)
        manager2.create_shared_array(f'test_{i}', arr)
        arrays.append(arr)
    
    print(f"  ✅ Created 10 shared arrays")
    
    # Verify all
    metadata2 = manager2.get_shared_arrays_metadata()
    all_ok = True
    shm_refs = []
    
    for i in range(10):
        meta = metadata2[f'test_{i}']
        attached, shm = attach_shared_array(
            meta['shm_name'],
            meta['shape'],
            meta['dtype']
        )
        shm_refs.append(shm)
        if not np.array_equal(arrays[i], attached):
            all_ok = False
            break
    
    if all_ok:
        print("  ✅ All arrays verified")
    else:
        print("  ❌ Some arrays mismatched!")
        return False
    
    # Cleanup
    for shm in shm_refs:
        shm.close()
    manager2.cleanup()
    print("  ✅ Cleanup successful")
    print()
    
    # Test 3: Memory usage
    print("Test 3: Memory usage monitoring")
    usage = get_memory_usage_mb()
    formatted = format_memory_usage(usage)
    print(f"  Current: {formatted}")
    print()
    
    print("=" * 60)
    print("🎉 All minimal tests passed!")
    print("=" * 60)
    print()
    print("Shared memory infrastructure is working correctly.")
    print("For full routing tests, ensure you have >8 GB available memory.")
    
    return True


if __name__ == "__main__":
    try:
        success = test_shared_memory_basic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
