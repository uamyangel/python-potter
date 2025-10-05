"""Utility helper functions."""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """Context manager to time code execution."""
    from .log import log

    start = time.time()
    log(f"{name} started...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        log(f"{name} completed in {elapsed:.2f}s")


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def format_size(size_bytes: int) -> str:
    """Format byte size in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
