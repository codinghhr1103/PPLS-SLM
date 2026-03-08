from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    log_dir: Union[str, Path],
    *,
    filename: str = "run.log",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    also_console: bool = True,
    force: bool = True,
) -> Path:
    """Configure process-wide Python logging.

    Returns the log file path.

    Notes:
    - We set `force=True` by default to make repeated calls predictable in notebooks / scripts.
    - Entry points should call this exactly once early.
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    if also_console:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=force)

    # Reduce noisy loggers from common numeric libs.
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_path
