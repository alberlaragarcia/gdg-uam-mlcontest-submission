"""v1 features: shortcut-free, offset- and scale-invariant summary of a (4, 256) sample.

Differences from v0 (experiment 20261110-features-v1, EDA §8):
  * removed: ``level``, per-channel mean / std, group one-hot, NaN fraction, mean log-std
    (train-only shortcuts or pure scale);
  * added for ch0 / ch1 (where the class lives): moments of the first differences and
    time-reversal asymmetry (class 5 is class 4 reversed in time), and the lag-8 ACF of the
    squared deviations (volatility clustering).

Layout (N_FEATURES = 4 * PER_CH + 2 * PER_SRC + 6 + 1 = 77):
  per channel (PER_CH = 15): relative log-std (vs the sample's mean log-std), skewness, excess
      kurtosis, ACF at ACF_LAGS (10), zero-crossing rate, std of first differences / std
  ch0, ch1 (PER_SRC = 5): dskew, dkurt (excess), trev1, trev3, acfsq8
  cross-channel: 6 Pearson correlations
  global: sample_rate
Same despike + NaN interpolation as v0 (``_fill``). stdlib + numpy + numba only.
"""

from __future__ import annotations

import numpy as np

from tsclf.features.v0 import ACF_LAGS, CLIP_Z, LENGTH, N_CH, SPIKE_Z, _fill, meta_scalars
from tsclf.inference.runtime import NJIT_OPTS, njit

PER_CH = 5 + len(ACF_LAGS)
PER_SRC = 5
N_SRC = 2  # ch0 and ch1 carry the class
N_FEATURES = N_CH * PER_CH + N_SRC * PER_SRC + 6 + 1
SQ_LAG = 8
TINY = 1e-12


TMP_SIZE = 2 * N_CH  # scratch of features_into: NaN fractions and log-stds


@njit(**NJIT_OPTS)
def compute_features(series, sample_rate, buf, out):
    """Fill ``out`` (N_FEATURES,) from ``series`` (4, 256). ``buf``: (4, 256) float64 scratch."""
    features_into(series, sample_rate, buf, out, np.empty(TMP_SIZE))


@njit(**NJIT_OPTS)
def features_into(series, sample_rate, buf, out, tmp):
    """``compute_features`` without allocations: ``tmp`` is a (TMP_SIZE,) float64 scratch."""
    C = series.shape[0]
    _fill(series, buf, tmp[:C])
    _from_filled(buf, sample_rate, out, tmp[C : 2 * C])


@njit(**NJIT_OPTS)
def features_fast(series, sample_rate, buf, out, tmp):
    """``features_into`` with ``fill_fast``: same features, cheaper cleaning (delivered path)."""
    fill_fast(series, buf)
    _from_filled(buf, sample_rate, out, tmp[: series.shape[0]])


@njit(**NJIT_OPTS)
def fill_fast(series, buf):
    """``tsclf.features.v0._fill`` (same rule) restructured for speed.

    The two statistics passes are branch-free sums (vectorisable); a plain copy counts the
    points to replace (non-finite or spikes); only if there are any, a last pass walks the bad
    runs and interpolates them. The NaN fraction is not computed: v1 does not use it.

    Same values as ``_fill`` on every input seen so far, with one known difference: ``_fill``
    squares float32 inputs in float32 and this function in float64. With a large offset
    relative to the spread (|mean| / sd >> 1e3, e.g. 1e4 + noise 0.01) and a spike, the
    rounding can move the spike threshold and one point may be interpolated here and kept
    there. Real samples have |mean| / sd <= 14 (val_shifted, test_demo).
    """
    C, L = series.shape
    for c in range(C):
        s1 = 0.0
        s2 = 0.0
        m = 0.0
        for t in range(L):
            v = np.float64(series[c, t])
            ok = np.isfinite(v)
            w = v if ok else 0.0
            s1 += w
            s2 += w * w
            m += 1.0 if ok else 0.0
        med = 0.0
        thr = np.inf
        if m > 4:
            mu = s1 / m
            lim = CLIP_Z * np.sqrt(max(s2 / m - mu * mu, 0.0))
            a1 = 0.0
            a2 = 0.0
            k = 0.0
            for t in range(L):
                v = np.float64(series[c, t])
                ok = np.isfinite(v) and abs(v - mu) <= lim
                w = v if ok else 0.0
                a1 += w
                a2 += w * w
                k += 1.0 if ok else 0.0
            if k > 4:
                med = a1 / k
                sd2 = np.sqrt(max(a2 / k - med * med, 0.0))
                if sd2 > 0.0:
                    thr = SPIKE_Z * sd2
        n_out = 0.0
        for t in range(L):
            v = np.float64(series[c, t])
            buf[c, t] = v
            d = abs(v - med)  # NaN for NaN, inf for +-inf: both fail the test below
            n_out += 0.0 if (d <= thr and d < np.inf) else 1.0
        if n_out == 0.0:
            continue
        last = -1
        t = 0
        while t < L:
            d = abs(buf[c, t] - med)
            if d <= thr and d < np.inf:
                last = t
                t += 1
                continue
            start = t
            while t < L:
                d = abs(buf[c, t] - med)
                if d <= thr and d < np.inf:
                    break
                t += 1
            if t == L:  # the run reaches the end: hold the last valid value (0 if none)
                v0 = buf[c, last] if last >= 0 else 0.0
                for u in range(start, L):
                    buf[c, u] = v0
            elif last == -1:  # the run starts the channel: take the first valid value
                v = buf[c, t]
                for u in range(t):
                    buf[c, u] = v
            else:
                v0 = buf[c, last]
                v = buf[c, t]
                for u in range(last + 1, t):
                    buf[c, u] = v0 + (v - v0) * (u - last) / (t - last)


@njit(**NJIT_OPTS)
def _from_filled(buf, sample_rate, out, logstd):
    """Features of a cleaned ``buf`` (overwritten with its z-scores); ``logstd``: (C,) scratch."""
    C, L = buf.shape
    mean_logstd = 0.0
    nl = len(ACF_LAGS)
    for c in range(C):
        s = 0.0
        for t in range(L):
            s += buf[c, t]
        m = s / L
        s2 = 0.0
        for t in range(L):
            d = buf[c, t] - m
            s2 += d * d
        sd = np.sqrt(s2 / L)
        logstd[c] = np.log(sd + 1e-6)
        mean_logstd += logstd[c] / C
        inv = 1.0 / sd if sd > 1e-9 else 0.0
        for t in range(L):  # z-score in place (buf is our own scratch)
            buf[c, t] = (buf[c, t] - m) * inv
    for c in range(C):
        o = c * PER_CH
        s3 = 0.0
        s4 = 0.0
        zc = 0
        dd = 0.0
        for t in range(L):
            z = buf[c, t]
            z2 = z * z
            s3 += z2 * z
            s4 += z2 * z2
            if t > 0:
                d = z - buf[c, t - 1]
                dd += d * d
                if (z > 0.0) != (buf[c, t - 1] > 0.0):
                    zc += 1
        m4 = s4 / L
        out[o + 0] = logstd[c]  # made relative below
        out[o + 1] = s3 / L
        out[o + 2] = m4 - 3.0
        for i in range(nl):
            lag = ACF_LAGS[i]
            a = 0.0
            for t in range(L - lag):
                a += buf[c, t] * buf[c, t + lag]
            out[o + 3 + i] = a / L
        out[o + 3 + nl] = zc / (L - 1)
        out[o + 4 + nl] = np.sqrt(dd / (L - 1))
        if c < N_SRC:
            p = C * PER_CH + c * PER_SRC
            # first differences: raw power sums -> central moments
            n1 = L - 1
            d1 = 0.0
            d2 = 0.0
            d3 = 0.0
            d4 = 0.0
            for t in range(1, L):
                d = buf[c, t] - buf[c, t - 1]
                e = d * d
                d1 += d
                d2 += e
                d3 += e * d
                d4 += e * e
            d1 /= n1
            d2 /= n1
            d3 /= n1
            d4 /= n1
            var = d2 - d1 * d1
            if var > TINY:
                c3 = d3 - 3.0 * d1 * d2 + 2.0 * d1 * d1 * d1
                c4 = d4 - 4.0 * d1 * d3 + 6.0 * d1 * d1 * d2 - 3.0 * d1 * d1 * d1 * d1
                out[p + 0] = c3 / (var * np.sqrt(var))
                out[p + 1] = c4 / (var * var) - 3.0
            else:
                out[p + 0] = 0.0
                out[p + 1] = 0.0
            out[p + 2] = d3 / (d2 * np.sqrt(d2)) if d2 > TINY else 0.0
            # time-reversal asymmetry at lag 3
            g2 = 0.0
            g3 = 0.0
            for t in range(3, L):
                d = buf[c, t] - buf[c, t - 3]
                e = d * d
                g2 += e
                g3 += e * d
            g2 /= L - 3
            g3 /= L - 3
            out[p + 3] = g3 / (g2 * np.sqrt(g2)) if g2 > TINY else 0.0
            # ACF of the squared deviations (z has mean 0 and variance 1: mean of z^2 is 1)
            vq = m4 - 1.0
            a = 0.0
            for t in range(L - SQ_LAG):
                u = buf[c, t]
                v = buf[c, t + SQ_LAG]
                a += (u * u - 1.0) * (v * v - 1.0)
            out[p + 4] = a / (L * vq) if vq > TINY else 0.0
    for c in range(C):
        out[c * PER_CH] -= mean_logstd
    o = C * PER_CH + N_SRC * PER_SRC
    k = 0
    for i in range(C):
        for j in range(i + 1, C):
            a = 0.0
            for t in range(L):
                a += buf[i, t] * buf[j, t]
            out[o + k] = a / L
            k += 1
    out[o + k] = sample_rate
    for i in range(out.shape[0]):  # last line of defence: never feed NaN/inf to the model
        if not np.isfinite(out[i]):
            out[i] = 0.0


def reference_features(x: np.ndarray, sample_rate: float) -> np.ndarray:
    """numpy twin for a clean (finite, spike-free) float64 ``(4, 256)`` sample (parity test)."""
    x = np.asarray(x, dtype=np.float64)
    L = x.shape[1]
    m, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True)
    z = (x - m) / sd
    logstd = np.log(sd[:, 0] + 1e-6)
    out = []
    for c in range(N_CH):
        zc = z[c]
        acf = [np.sum(zc[:-lag] * zc[lag:]) / L for lag in ACF_LAGS]
        out += [logstd[c] - logstd.mean(), np.mean(zc**3), np.mean(zc**4) - 3.0, *acf]
        out += [np.mean((zc[1:] > 0) != (zc[:-1] > 0)), np.sqrt(np.mean(np.diff(zc) ** 2))]
    for c in range(N_SRC):
        zc = z[c]
        d = np.diff(zc)
        dc = d - d.mean()
        d3 = zc[3:] - zc[:-3]
        q = zc**2 - 1.0
        out += [
            np.mean(dc**3) / np.mean(dc**2) ** 1.5,
            np.mean(dc**4) / np.mean(dc**2) ** 2 - 3.0,
            np.mean(d**3) / np.mean(d**2) ** 1.5,
            np.mean(d3**3) / np.mean(d3**2) ** 1.5,
            np.sum(q[:-SQ_LAG] * q[SQ_LAG:]) / (L * np.mean(q**2)),
        ]
    out += [np.mean(z[i] * z[j]) for i in range(N_CH) for j in range(i + 1, N_CH)]
    return np.array([*out, sample_rate])


def features_batch(series: np.ndarray, meta: dict) -> np.ndarray:
    """(n, 4, 256) series + meta arrays -> (n, N_FEATURES) float32. Training-time helper."""
    n = len(series)
    X = np.empty((n, N_FEATURES), dtype=np.float32)
    buf = np.empty((N_CH, LENGTH))
    out = np.empty(N_FEATURES)
    sr = meta["sample_rate"]
    for j in range(n):
        s, _, _ = meta_scalars({"sample_rate": sr[j]})
        compute_features(np.asarray(series[j]), s, buf, out)
        X[j] = out
    return X
