import numpy as np


def random_stiefel(rows: int, cols: int, rng: np.random.RandomState) -> np.ndarray:
    q, _ = np.linalg.qr(rng.randn(rows, cols))
    return q[:, :cols]


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm(np.asarray(a) - np.asarray(b))
    den = max(1.0, np.linalg.norm(np.asarray(a)), np.linalg.norm(np.asarray(b)))
    return float(num / den)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
