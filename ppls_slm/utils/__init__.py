"""Small utility helpers used across entry points.

This subpackage intentionally stays dependency-free.
"""

from .logging_utils import setup_logging
from .paths import ensure_dir, repo_root, resolve_path
from .random_utils import set_global_seed

__all__ = [
    "ensure_dir",
    "repo_root",
    "resolve_path",
    "set_global_seed",
    "setup_logging",
]

