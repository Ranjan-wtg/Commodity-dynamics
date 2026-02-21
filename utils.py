"""
utils.py — Shared utilities: seeding, logging, serialization.
"""

import os
import random
import logging
import joblib
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def save_artifact(obj, path: str) -> None:
    """Serialize object to disk using joblib."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(obj, path)
    get_logger("utils").info(f"Saved artifact → {path}")


def load_artifact(path: str):
    """Load serialized artifact from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path
