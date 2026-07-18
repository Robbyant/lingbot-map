"""Logging utilities for the benchmark framework.

Provides unified logging setup.
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: Path,
    name: str = 'benchmark',
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """Setup unified logging system with file and console handlers.

    Args:
        log_dir: Directory for log files
        name: Logger name (e.g., 'prepare', 'run')
        console_level: Console logging level (default: INFO)
        file_level: File logging level (default: DEBUG)

    Returns:
        Configured logger instance

    Note:
        - File logs include full debug information with timestamps
        - Console logs show only specified level and above
        - Log files are named with timestamp: {name}_YYYYMMDD_HHMMSS.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler - full format, all information
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Console handler - key information only
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

    return logger
