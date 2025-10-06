"""
Memory usage monitoring utilities.

Provides cross-platform memory usage tracking for both the current process
and system-wide memory statistics.
"""

import os
import sys
from typing import Dict, Optional


def get_memory_usage_mb() -> Dict[str, float]:
    """
    Get current process memory usage in MB.
    
    Returns:
        Dictionary with keys:
        - 'rss': Resident Set Size (physical memory)
        - 'vms': Virtual Memory Size (total memory)
        - 'available': Available system memory (if available)
    """
    usage = {}
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        # Process memory
        usage['rss'] = mem_info.rss / 1024 / 1024  # bytes to MB
        usage['vms'] = mem_info.vms / 1024 / 1024
        
        # System memory
        sys_mem = psutil.virtual_memory()
        usage['available'] = sys_mem.available / 1024 / 1024
        usage['percent'] = sys_mem.percent
        
    except ImportError:
        # Fallback: try resource module (Unix-like systems)
        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is in KB on Linux, bytes on macOS
            if sys.platform == 'darwin':
                usage['rss'] = rusage.ru_maxrss / 1024 / 1024
            else:
                usage['rss'] = rusage.ru_maxrss / 1024
            usage['vms'] = 0.0
            usage['available'] = 0.0
            usage['percent'] = 0.0
        except Exception:
            # Last resort: return zeros
            usage = {'rss': 0.0, 'vms': 0.0, 'available': 0.0, 'percent': 0.0}
    
    return usage


def format_memory_usage(usage: Dict[str, float]) -> str:
    """
    Format memory usage dictionary as human-readable string.
    
    Args:
        usage: Dictionary from get_memory_usage_mb()
        
    Returns:
        Formatted string like "RSS: 1234.5 MB, Available: 8192.0 MB (12.3%)"
    """
    parts = []
    
    if usage.get('rss', 0) > 0:
        parts.append(f"RSS: {usage['rss']:.1f} MB")
    
    if usage.get('vms', 0) > 0:
        parts.append(f"VMS: {usage['vms']:.1f} MB")
    
    if usage.get('available', 0) > 0:
        parts.append(f"Available: {usage['available']:.1f} MB")
    
    if usage.get('percent', 0) > 0:
        parts.append(f"({usage['percent']:.1f}% used)")
    
    return ", ".join(parts) if parts else "Memory info unavailable"


def log_memory_usage(prefix: str = ""):
    """
    Log current memory usage.
    
    Args:
        prefix: Optional prefix for the log message
    """
    from .log import log
    usage = get_memory_usage_mb()
    formatted = format_memory_usage(usage)
    log(f"{prefix}{formatted}")


def check_memory_available(required_mb: float) -> bool:
    """
    Check if enough memory is available.
    
    Args:
        required_mb: Required memory in MB
        
    Returns:
        True if enough memory is available
    """
    usage = get_memory_usage_mb()
    available = usage.get('available', float('inf'))
    return available >= required_mb


def estimate_multiprocessing_memory(
    base_memory_mb: float,
    num_processes: int,
    shared_memory_mb: float = 0.0
) -> Dict[str, float]:
    """
    Estimate memory usage for multiprocessing scenario.
    
    Args:
        base_memory_mb: Base memory usage (single process)
        num_processes: Number of worker processes
        shared_memory_mb: Shared memory size (not duplicated)
        
    Returns:
        Dictionary with estimated memory breakdown
    """
    # Each worker process needs its own copy (minus shared memory)
    per_process = base_memory_mb - shared_memory_mb
    
    # Total = main process + workers + shared memory (once)
    total = base_memory_mb + (per_process * num_processes) + shared_memory_mb
    
    return {
        'base_mb': base_memory_mb,
        'shared_mb': shared_memory_mb,
        'per_worker_mb': per_process,
        'num_workers': num_processes,
        'total_estimated_mb': total,
        'savings_from_sharing_mb': shared_memory_mb * num_processes
    }


def format_memory_estimate(estimate: Dict[str, float]) -> str:
    """
    Format memory estimate as human-readable string.
    
    Args:
        estimate: Dictionary from estimate_multiprocessing_memory()
        
    Returns:
        Multi-line formatted string
    """
    lines = [
        f"Memory Estimate (with {estimate['num_workers']} workers):",
        f"  Base process: {estimate['base_mb']:.1f} MB",
        f"  Shared memory: {estimate['shared_mb']:.1f} MB (zero-copy)",
        f"  Per worker: {estimate['per_worker_mb']:.1f} MB",
        f"  Total estimated: {estimate['total_estimated_mb']:.1f} MB",
        f"  Savings from sharing: {estimate['savings_from_sharing_mb']:.1f} MB"
    ]
    return "\n".join(lines)
