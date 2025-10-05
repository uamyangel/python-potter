"""Logging utilities."""

import sys
from datetime import datetime
from typing import TextIO


class Logger:
    """Simple logger for Potter."""

    def __init__(self, file: TextIO = sys.stdout):
        self.file = file
        self.verbose = True

    def __call__(self, *args, **kwargs) -> None:
        """Log a message."""
        if not self.verbose:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = " ".join(str(arg) for arg in args)
        print(f"[{timestamp}] {message}", file=self.file, **kwargs)
        self.file.flush()


# Global logger instance
_logger = Logger()


def log(*args, **kwargs) -> Logger:
    """Get logger instance and optionally log a message."""
    if args or kwargs:
        _logger(*args, **kwargs)
    return _logger


def set_log_file(filepath: str) -> None:
    """Set log output file."""
    global _logger
    _logger.file = open(filepath, 'w')


def set_verbose(verbose: bool) -> None:
    """Set verbose logging."""
    _logger.verbose = verbose
