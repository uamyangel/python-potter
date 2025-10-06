"""
Shared memory manager for zero-copy multiprocessing.

FIRST PRINCIPLES:
- NumPy arrays are the largest data (hundreds of MB)
- multiprocessing.shared_memory allows zero-copy sharing across processes
- Only the memory NAME (string) needs to be pickled, not the data itself
- Memory usage: O(1) instead of O(num_processes)

This module manages the lifecycle of shared memory blocks for routing graph data.
"""

import numpy as np
from multiprocessing import shared_memory
from typing import Dict, Tuple, Optional
import atexit


class SharedMemoryManager:
    """Manages shared memory blocks for zero-copy inter-process communication."""
    
    def __init__(self):
        """Initialize the shared memory manager."""
        self.shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self.array_metadata: Dict[str, Tuple] = {}  # name -> (shape, dtype)
        
    def create_shared_array(
        self, 
        name: str, 
        source_array: np.ndarray
    ) -> str:
        """
        Create a shared memory block from a NumPy array.
        
        Args:
            name: Unique name for this shared array
            source_array: Source NumPy array to share
            
        Returns:
            Shared memory block name (for passing to worker processes)
        """
        # Create shared memory block
        shm = shared_memory.SharedMemory(
            create=True, 
            size=source_array.nbytes,
            name=None  # Let system assign unique name
        )
        
        # Create NumPy array view on shared memory
        shared_array = np.ndarray(
            source_array.shape,
            dtype=source_array.dtype,
            buffer=shm.buf
        )
        
        # Copy data to shared memory
        shared_array[:] = source_array[:]
        
        # Store metadata for later cleanup
        self.shm_blocks[name] = shm
        self.array_metadata[name] = (source_array.shape, source_array.dtype)
        
        return shm.name  # Return system-assigned name
    
    def get_shared_arrays_metadata(self) -> Dict[str, Dict]:
        """
        Get metadata for all shared arrays (for passing to workers).
        
        Returns:
            Dictionary mapping array name to {shm_name, shape, dtype}
        """
        metadata = {}
        for name, shm in self.shm_blocks.items():
            shape, dtype = self.array_metadata[name]
            metadata[name] = {
                'shm_name': shm.name,
                'shape': shape,
                'dtype': str(dtype),  # Convert to string for pickling
            }
        return metadata
    
    def cleanup(self):
        """Clean up all shared memory blocks."""
        for name, shm in self.shm_blocks.items():
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                # Ignore cleanup errors (memory may already be freed)
                pass
        self.shm_blocks.clear()
        self.array_metadata.clear()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def attach_shared_array(
    shm_name: str,
    shape: Tuple,
    dtype_str: str
) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    """
    Attach to an existing shared memory block and return NumPy array view.
    
    This function is called in worker processes to access shared arrays.
    
    IMPORTANT: Returns both the array AND the SharedMemory object.
    The caller MUST keep the SharedMemory object alive, or the array will
    become invalid and cause segmentation faults.
    
    Args:
        shm_name: System-assigned shared memory name
        shape: Array shape
        dtype_str: Data type as string
        
    Returns:
        Tuple of (numpy_array, shared_memory_object)
        The caller must store the shared_memory_object to keep it alive!
    """
    # Attach to existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    
    # Create NumPy array view
    dtype = np.dtype(dtype_str)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    # CRITICAL: Return both array and shm object!
    # If we don't return shm, it gets garbage collected and the array becomes invalid
    return array, shm


# Global manager for cleanup on exit
_global_manager: Optional[SharedMemoryManager] = None


def get_global_manager() -> SharedMemoryManager:
    """Get or create the global shared memory manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = SharedMemoryManager()
        # Register cleanup on exit
        atexit.register(_global_manager.cleanup)
    return _global_manager


def cleanup_global_manager():
    """Clean up the global shared memory manager."""
    global _global_manager
    if _global_manager is not None:
        _global_manager.cleanup()
        _global_manager = None
