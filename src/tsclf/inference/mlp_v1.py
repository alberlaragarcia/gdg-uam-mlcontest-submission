"""v1 inference kernel: v1 features -> MLP (1 hidden ReLU layer) -> argmax, in ONE numba call.

Standardization is folded into W1/b1 and the logit adjustment + decision biases into b2 at export
time (scripts/train_v1.py), so the decision is a plain argmax. stdlib + numpy + numba only.
"""

from __future__ import annotations

import numpy as np

from tsclf.features.v1 import (
    LENGTH,
    N_CH,
    N_FEATURES,
    TMP_SIZE,
    compute_features,
    features_fast,
    features_into,
)
from tsclf.inference.runtime import NJIT_OPTS, njit


@njit(**NJIT_OPTS)
def predict_v1(series, sample_rate, buf, feat, hid, W1, b1, W2, b2):
    compute_features(series, sample_rate, buf, feat)
    F, H = W1.shape
    for j in range(H):
        hid[j] = b1[j]
    for i in range(F):
        f = feat[i]
        for j in range(H):
            hid[j] += f * W1[i, j]
    best, best_v = 0, -np.inf
    for k in range(W2.shape[1]):
        s = b2[k]
        for j in range(H):
            h = hid[j]
            if h > 0.0:
                s += h * W2[j, k]
        if s > best_v:
            best, best_v = k, s
    return best


def work_size(hidden: int, n_classes: int = 8) -> int:
    """Length of the float64 work array of ``make_predictor``'s kernel."""
    return N_CH * LENGTH + N_FEATURES + hidden + TMP_SIZE + n_classes


def make_predictor(W1, b1, W2, b2, fast: bool = True, float32_weights: bool = False):
    """Kernel ``predict(series, sample_rate, work) -> int`` with the weights frozen as constants.

    Same decisions as ``predict_v1``, but the call has 3 arguments instead of 9 (numba checks
    and unboxes every argument on each call) and nothing is allocated: ``work`` is one
    preallocated float64 array of ``work_size(hidden)`` elements. A non-finite ``sample_rate``
    is replaced by the neutral 1.5, as ``tsclf.features.v0.meta_scalars`` does.

    ``fast`` (default) uses ``features_fast`` and an output layer that goes hidden unit by
    hidden unit and skips the inactive ones; ``fast=False`` is the kernel delivered before
    experiment 20261110-latency-kernel (reference for A/B tests). ``float32_weights`` stores
    ``W1``/``W2`` in float32 (half the memory; the arithmetic stays in float64).
    """
    wtype = np.float32 if float32_weights else np.float64
    W1 = np.ascontiguousarray(W1, dtype=wtype)
    b1 = np.ascontiguousarray(b1, dtype=np.float64)
    W2 = np.ascontiguousarray(W2, dtype=wtype)
    b2 = np.ascontiguousarray(b2, dtype=np.float64)
    F, H = W1.shape
    K = W2.shape[1]
    n_buf = N_CH * LENGTH
    o_hid = n_buf + F
    o_tmp = o_hid + H
    o_acc = o_tmp + TMP_SIZE

    @njit(**NJIT_OPTS)
    def predict_ref(series, sample_rate, work):
        if not np.isfinite(sample_rate):
            sample_rate = 1.5
        buf = work[:n_buf].reshape(N_CH, LENGTH)
        feat = work[n_buf:o_hid]
        hid = work[o_hid:o_tmp]
        features_into(series, sample_rate, buf, feat, work[o_tmp:o_acc])
        for j in range(H):
            hid[j] = b1[j]
        for i in range(F):
            f = feat[i]
            for j in range(H):
                hid[j] += f * W1[i, j]
        best, best_v = 0, -np.inf
        for k in range(K):
            s = b2[k]
            for j in range(H):
                h = hid[j]
                if h > 0.0:
                    s += h * W2[j, k]
            if s > best_v:
                best, best_v = k, s
        return best

    @njit(**NJIT_OPTS)
    def predict_fast(series, sample_rate, work):
        if not np.isfinite(sample_rate):
            sample_rate = 1.5
        buf = work[:n_buf].reshape(N_CH, LENGTH)
        feat = work[n_buf:o_hid]
        hid = work[o_hid:o_tmp]
        acc = work[o_acc : o_acc + K]
        features_fast(series, sample_rate, buf, feat, work[o_tmp:o_acc])
        for j in range(H):
            hid[j] = b1[j]
        for i in range(F):
            f = feat[i]
            for j in range(H):
                hid[j] += f * W1[i, j]
        for k in range(K):
            acc[k] = b2[k]
        for j in range(H):
            h = hid[j]
            if h > 0.0:
                for k in range(K):
                    acc[k] += h * W2[j, k]
        best, best_v = 0, -np.inf
        for k in range(K):
            if acc[k] > best_v:
                best, best_v = k, acc[k]
        return best

    return predict_fast if fast else predict_ref
