"""v0 features: NaN-safe, mostly scale/offset-invariant summary of a (4, 256) sample.

One numba kernel fills a preallocated float64 vector. Used for training AND inference, so the
feature definition cannot drift between the two. stdlib + numpy + numba only.

Despike + NaN interpolation first (``_fill``). Layout (N_FEATURES = 4 * PER_CH + 6 + 6):
  per channel (PER_CH = 17): nan fraction, relative log-std (vs the sample's mean log-std),
      mean/std, skewness, excess kurtosis, ACF at ACF_LAGS (10), zero-crossing rate,
      std of first differences / std
  cross-channel: 6 Pearson correlations
  global: mean log-std, sample_rate, level, group one-hot (A, B, C)
"""

from __future__ import annotations

import numpy as np

from tsclf.inference.runtime import NJIT_OPTS, njit

N_CH, LENGTH = 4, 256
ACF_LAGS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32], dtype=np.int64)
PER_CH = 7 + len(ACF_LAGS)
N_FEATURES = N_CH * PER_CH + 6 + 6
GROUPS = ("A", "B", "C")


SPIKE_Z = 7.0  # |x - m2| > SPIKE_Z * s2 -> spike (2-pass sigma clipping; train p99.99 ~ 7)
CLIP_Z = 6.0  # first pass: points beyond CLIP_Z raw std are left out of (m2, s2)


@njit(**NJIT_OPTS)
def _fill(series, buf, nanfrac):
    """Copy ``series`` into ``buf`` with non-finite values and isolated spikes interpolated.

    Spikes (|x - m2| > SPIKE_Z * s2, with m2/s2 the mean/std after a first 6-sigma clip) do not
    occur in train but hit ~15 % of test samples; this catches all of them on test_demo and
    flags ~1e-4 of train channels. Sort-free (a median-based rule cost ~25 us per sample).
    Edges take the nearest valid value; a channel with no valid value becomes 0.
    ``nanfrac`` gets the fraction of non-finite values (spikes are not counted).
    """
    C, L = series.shape
    for c in range(C):
        s1, s2, m = 0.0, 0.0, 0
        for t in range(L):
            v = series[c, t]
            if np.isfinite(v):
                s1 += v
                s2 += v * v
                m += 1
        med, thr = 0.0, np.inf
        if m > 4:
            mu = s1 / m
            sd = np.sqrt(max(s2 / m - mu * mu, 0.0))
            a1, a2, k = 0.0, 0.0, 0
            for t in range(L):
                v = series[c, t]
                if np.isfinite(v) and abs(v - mu) <= CLIP_Z * sd:
                    a1 += v
                    a2 += v * v
                    k += 1
            if k > 4:
                med = a1 / k
                sd2 = np.sqrt(max(a2 / k - med * med, 0.0))
                if sd2 > 0.0:
                    thr = SPIKE_Z * sd2
        last = -1
        n_bad = 0
        for t in range(L):
            v = series[c, t]
            finite = np.isfinite(v)
            if not finite:
                n_bad += 1
            if finite and abs(v - med) <= thr:
                if last == -1:
                    for u in range(t):
                        buf[c, u] = v
                elif t - last > 1:
                    v0 = buf[c, last]
                    for u in range(last + 1, t):
                        buf[c, u] = v0 + (v - v0) * (u - last) / (t - last)
                buf[c, t] = v
                last = t
        if last == -1:
            for u in range(L):
                buf[c, u] = 0.0
        else:
            for u in range(last + 1, L):
                buf[c, u] = buf[c, last]
        nanfrac[c] = n_bad / L


@njit(**NJIT_OPTS)
def compute_features(series, sample_rate, level, group_code, buf, out):
    """Fill ``out`` (N_FEATURES,) from ``series`` (4, 256). ``buf``: (4, 256) float64 scratch."""
    C, L = series.shape
    nanfrac = np.empty(C)
    _fill(series, buf, nanfrac)
    means = np.empty(C)
    stds = np.empty(C)
    for c in range(C):
        s = 0.0
        for t in range(L):
            s += buf[c, t]
        m = s / L
        s2 = 0.0
        for t in range(L):
            d = buf[c, t] - m
            s2 += d * d
        means[c] = m
        stds[c] = np.sqrt(s2 / L)
    logstd = np.empty(C)
    mean_logstd = 0.0
    for c in range(C):
        logstd[c] = np.log(stds[c] + 1e-6)
        mean_logstd += logstd[c] / C
    nl = len(ACF_LAGS)
    for c in range(C):
        o = c * (7 + nl)
        sd = stds[c]
        inv = 1.0 / sd if sd > 1e-9 else 0.0
        m = means[c]
        # z-score in place (buf is our own scratch)
        for t in range(L):
            buf[c, t] = (buf[c, t] - m) * inv
        s3 = 0.0
        s4 = 0.0
        zc = 0
        dd = 0.0
        for t in range(L):
            z = buf[c, t]
            s3 += z * z * z
            s4 += z * z * z * z
            if t > 0:
                d = z - buf[c, t - 1]
                dd += d * d
                if (z > 0.0) != (buf[c, t - 1] > 0.0):
                    zc += 1
        out[o + 0] = nanfrac[c]
        out[o + 1] = logstd[c] - mean_logstd
        out[o + 2] = m * inv
        out[o + 3] = s3 / L
        out[o + 4] = s4 / L - 3.0
        for i in range(nl):
            lag = ACF_LAGS[i]
            a = 0.0
            for t in range(L - lag):
                a += buf[c, t] * buf[c, t + lag]
            out[o + 5 + i] = a / L
        out[o + 5 + nl] = zc / (L - 1)
        out[o + 6 + nl] = np.sqrt(dd / (L - 1))
    o = C * (7 + nl)
    k = 0
    for i in range(C):
        for j in range(i + 1, C):
            a = 0.0
            for t in range(L):
                a += buf[i, t] * buf[j, t]
            out[o + k] = a / L
            k += 1
    o += k
    out[o + 0] = mean_logstd
    out[o + 1] = sample_rate
    out[o + 2] = level
    for g in range(3):
        out[o + 3 + g] = 1.0 if group_code == g else 0.0
    for i in range(out.shape[0]):  # last line of defence: never feed NaN/inf to the model
        if not np.isfinite(out[i]):
            out[i] = 0.0


def meta_scalars(meta) -> tuple[float, float, int]:
    """(sample_rate, level, group_code) from a meta dict; unknown or missing values -> neutral."""
    try:
        sr = float(meta.get("sample_rate", 1.5))
    except (TypeError, ValueError):
        sr = 1.5
    try:
        lv = float(meta.get("level", 0.0))
    except (TypeError, ValueError):
        lv = 0.0
    if not np.isfinite(sr):
        sr = 1.5
    if not np.isfinite(lv):
        lv = 0.0
    lv = min(max(lv, -10.0), 10.0)
    g = meta.get("group")
    gc = GROUPS.index(g) if g in GROUPS else -1
    return sr, lv, gc


def features_batch(series: np.ndarray, meta: dict) -> np.ndarray:
    """(n, 4, 256) series + meta arrays -> (n, N_FEATURES) float32. Training-time helper."""
    n = len(series)
    X = np.empty((n, N_FEATURES), dtype=np.float32)
    buf = np.empty((N_CH, LENGTH))
    out = np.empty(N_FEATURES)
    sr, lv, gr = meta["sample_rate"], meta["level"], meta["group"]
    for j in range(n):
        s, lvl, g = meta_scalars({"sample_rate": sr[j], "level": lv[j], "group": str(gr[j])})
        compute_features(np.asarray(series[j]), s, lvl, g, buf, out)
        X[j] = out
    return X
