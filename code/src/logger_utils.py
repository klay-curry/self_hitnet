from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple


def get_timestamp() -> str:
    """Return compact timestamp suitable for filenames (YYYYMMDD_HHMM)."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def setup_logger(log_dir: Path, label: str, timestamp: str | None = None) -> Tuple[logging.Logger, str]:
    """Create a logger that writes to file and stdout.

    Args:
        log_dir: Directory to place log files.
        label: Log label, e.g., "train" or "interface".
        timestamp: Optional fixed timestamp; if None, generate.
    Returns:
        (logger, timestamp) used for the filename.
    """
    if timestamp is None:
        timestamp = get_timestamp()

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{timestamp}_{label}.log"

    logger = logging.getLogger(f"cod_{label}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, timestamp
