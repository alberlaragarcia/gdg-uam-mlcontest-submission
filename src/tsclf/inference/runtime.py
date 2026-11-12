"""Inference runtime: numba kernels and their warmup. stdlib + numpy + numba only."""

from __future__ import annotations

import time

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # numba missing or broken in the evaluation env: slow but still scores
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in for numba.njit: the kernels run as plain Python/numpy."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


# Never fastmath=True: it implies nnan/ninf and silently breaks NaN handling.
NJIT_OPTS = {"cache": False, "nogil": True, "fastmath": {"contract", "arcp", "reassoc"}}

N_CHANNELS, LENGTH = 4, 256


@njit(**NJIT_OPTS)
def nan_safe_mean(series: np.ndarray) -> float:
    """Mean of the finite values of a (C, L) float32 array (0.0 if there are none)."""
    s, n = 0.0, 0
    for c in range(series.shape[0]):
        for t in range(series.shape[1]):
            v = series[c, t]
            if np.isfinite(v):
                s += v
                n += 1
    return s / n if n else 0.0


def synthetic_samples() -> list[np.ndarray]:
    """Read-only (4, 256) float32 inputs covering the code paths: normal, partial and all NaN."""
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(N_CHANNELS, LENGTH)).astype(np.float32)
    gaps = normal.copy()
    gaps[1, 40:90] = np.nan
    gaps[3, 0] = np.inf
    all_nan = np.full((N_CHANNELS, LENGTH), np.nan, dtype=np.float32)
    out = [normal, gaps, all_nan]
    for a in out:
        a.setflags(write=False)
    return out


def warmup(kernels=(nan_safe_mean,), repeats: int = 3) -> float:
    """Compile and exercise every kernel on the synthetic samples; returns seconds spent."""
    t0 = time.perf_counter()
    for x in synthetic_samples():
        for kernel in kernels:
            for _ in range(repeats):
                kernel(x)
    return time.perf_counter() - t0
