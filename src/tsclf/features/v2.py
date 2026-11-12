"""v2 features: a numba kernel GENERATED from a feature specification (``config.json["features"]``).

The specification is an ordered list of primitives ``{"op": ..., "ch": [...], <params>}`` (the
contract is ``docs/notes/feature_primitives_v2.md``). Every parameter becomes a literal in the
generated source, so lags, dilations, channels and biases are compile-time constants: with
runtime arrays LLVM does not vectorise and the ACF alone costs ~4x more
(``docs/notes/latency_prices.md``). stdlib + numpy + numba only (inference path).

Conventions (identical to ``tsclf.features.v1``):
  * each used channel is cleaned (``fill_channel`` = ``fill_fast`` restricted to one channel),
    then z-normalised with its mean and population sd (divisor L); a constant channel gives z = 0;
    ``logstd = log(sd + 1e-6)``;
  * ``rel_logstd``  logstd[c] - mean(logstd[ref_ch]);
  * ``moments``     skew = mean(z^3), exkurt = mean(z^4) - 3;
  * ``zcr``         sign changes of z / (L - 1);  ``ddstd``  sqrt(sum(diff(z)^2) / (L - 1));
  * ``acf``         sum_{t < L-lag} z_t z_{t+lag} / L, one output per lag;
  * ``diffmom``     on d = diff(z): dskew and dkurt (central moments, divisor L - 1; 0 if the
                    variance is <= 1e-12) and trev1 = mean(d^3) / mean(d^2)^1.5;
  * ``trev``        d = z_t - z_{t-lag} (divisor L - lag): mean(d^3) / mean(d^2)^1.5;
  * ``acfsq``       sum_{t < L-lag} (z_t^2 - 1)(z_{t+lag}^2 - 1) / (L (mean(z^4) - 1));
  * ``corr``        sum_t z_i z_j / L per pair;
  * ``dcorr``/``dacf`` on the STANDARDISED first differences of z (mean 0, sd 1, divisor
                    L - 1; a constant difference series gives 0): sum / (L - 1);
  * ``minirocket``  PPV of MiniRocket columns on z (``tsclf.models.minirocket`` definition):
                    C[t] = 3 (z[t+(a-4)d] + z[t+(b-4)d] + z[t+(c-4)d]) - sum_k z[t+(k-4)d] with
                    zero padding; fraction of C > bias over all L positions (padded) or the
                    L - 8d valid ones. The same columns are applied to every channel in ``ch``.
                    The bias must be fitted on THIS z (mean/sd, ``zscore_batch``): the biases
                    of ``tsclf.models.minirocket.fit_biases`` use median/MAD and do not transfer;
  * ``conv_relu_mean`` one learned 9-tap filter on z (PyTorch ``Conv1d`` semantics: no flip, no
                    padding): v_i = bias + sum_k taps[k] z[i*stride + k*dilation] for every
                    valid output i (i*stride + 8*dilation < L); output = mean_i max(v_i, 0). One
                    output per channel in ``ch`` (the filter is shared between channels);
  * virtual channel ``"s01"`` (accepted in ``ch`` / ``pairs`` of moments, zcr, ddstd, acf,
                    diffmom, trev, acfsq, corr, minirocket, conv_relu_mean, volwin, env; not in
                    rel_logstd, dcorr, dacf): the common source of ch0/ch1,
                    s = (z0 + z1 - mean) / sd(z0 + z1) (population sd; 0 if sd <= 1e-12), i.e.
                    (z0 + z1) / sqrt(2 + 2 rho01) for non-constant channels. Built once per
                    call into ``buf[4]`` (``buf_rows(spec)`` = 5 rows), in one pass that also
                    accumulates ``corr`` (0, 1) when the spec has it;
  * ``volwin``      per channel, 4 outputs: CV (sd / mean, mean floored at 1e-12) of the means
                    of z^2 over non-overlapping windows of 8, 16 and 32, then
                    log((max + 1e-3) / (min + 1e-3)) of the 32-window means;
  * ``env``         per channel, 2 outputs: m = 3-point moving average of z (valid), u = m[1:-1]
                    and v = m[2:] - m[:-2] standardised (sd floored at 1e-12), R = u^2 + v^2:
                    mean(sqrt R) / sqrt(mean R) and sd(R) / mean R (mean R floored at 1e-12);
  * ``sample_rate`` the meta value (non-finite -> 1.5);
  * outputs are op by op in specification order, channel by channel inside an op; any
    non-finite output becomes 0.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations

import numpy as np

from tsclf.features.v0 import CLIP_Z, SPIKE_Z
from tsclf.inference.runtime import NJIT_OPTS, njit

N_CH, LENGTH = 4, 256
TINY = 1e-12
KLEN = 9
MAX_FUSED_FILTERS = 4  # conv_relu_mean filters per fused loop (gain measured at 4)
MAX_DILATION = (LENGTH - 1) // 8  # 8d < L keeps at least one valid MiniRocket position
AUX_SHAPE = (N_CH + 1, LENGTH + 8 * MAX_DILATION)  # rows 0-3: std. diffs; row 4: padded z
MR_INDICES = tuple(combinations(range(KLEN), 3))  # the 84 MiniRocket kernels, same order
NEUTRAL_SAMPLE_RATE = 1.5

# op -> (required params, optional params with defaults)
OPS: dict[str, tuple[tuple[str, ...], dict]] = {
    "fill": (("ch",), {}),
    "znorm": (("ch",), {}),
    "rel_logstd": (("ch",), {"ref_ch": [0, 1, 2, 3]}),
    "moments": (("ch",), {}),
    "zcr": (("ch",), {}),
    "ddstd": (("ch",), {}),
    "acf": (("ch", "lags"), {}),
    "diffmom": (("ch",), {}),
    "trev": (("ch", "lag"), {}),
    "acfsq": (("ch", "lag"), {}),
    "corr": (("pairs",), {}),
    "dcorr": (("pairs",), {}),
    "dacf": (("ch", "lags"), {}),
    "minirocket": (("ch", "dilation", "cols"), {}),
    "conv_relu_mean": (("ch", "dilation", "taps", "bias"), {"stride": 1}),
    "volwin": (("ch",), {}),
    "env": (("ch",), {}),
    "c3": (("ch", "lags"), {}),
    "m3": (("ch", "pairs"), {}),
    "envacf": (("ch", "lags"), {}),
    "sample_rate": ((), {}),
}
# Virtual channels: names accepted in ``ch`` (and ``corr`` pairs) of the ops below; the kernel
# builds them once per call into the extra rows of ``buf`` (row N_CH + i for VIRTUAL[i]).
VIRTUAL = ("s01",)
VIRTUAL_OPS = frozenset(
    {
        "moments",
        "zcr",
        "ddstd",
        "acf",
        "diffmom",
        "trev",
        "acfsq",
        "corr",
        "minirocket",
        "conv_relu_mean",
        "volwin",
        "env",
        "c3",
        "m3",
        "envacf",
    }
)
# ops that never write the scratch row dz[N_CH]: the envelope R that ``env`` leaves there stays
# valid across them for a following ``envacf`` on the same channel
_KEEPS_ENV_ROW = frozenset(
    {"rel_logstd", "moments", "zcr", "ddstd", "diffmom", "trev", "corr", "dcorr", "dacf", "c3"}
)
VOLWIN_W = (8, 16, 32)  # window lengths of ``volwin`` (each a doubling of the previous)


class SpecError(ValueError):
    """A feature specification the generator cannot build."""


# ----------------------------------------------------------------------------- specification


def _channels(v, name: str, virtual: bool = False) -> list:
    """Physical channels (ints 0..3) and, if ``virtual``, the names in ``VIRTUAL``."""
    if not isinstance(v, list | tuple) or any(
        not (isinstance(c, int) and not isinstance(c, bool)) and not (virtual and c in VIRTUAL)
        for c in v
    ):
        allowed = f"ints or {list(VIRTUAL)}" if virtual else "ints"
        raise SpecError(f"{name} must be a list of {allowed}, got {v!r}")
    if any(isinstance(c, int) and not 0 <= c < N_CH for c in v):
        raise SpecError(f"{name}: channels must be in 0..{N_CH - 1}, got {v!r}")
    return [c if isinstance(c, str) else int(c) for c in v]


def _lag(v, name: str, hi: int) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= hi:
        raise SpecError(f"{name} must be an int in 1..{hi}, got {v!r}")
    return int(v)


def normalize_spec(spec) -> list[dict]:
    """Validated, canonical copy of ``spec`` (lists, ints, floats). Raises ``SpecError``."""
    if not isinstance(spec, list | tuple):
        raise SpecError("the feature specification must be a list of primitives")
    out = []
    for i, item in enumerate(spec):
        if not isinstance(item, dict) or "op" not in item:
            raise SpecError(f"item {i}: expected a dict with 'op', got {item!r}")
        op = item["op"]
        if op not in OPS:
            raise SpecError(f"item {i}: unknown op {op!r} (known: {sorted(OPS)})")
        required, optional = OPS[op]
        unknown = set(item) - {"op", *required, *optional}
        missing = set(required) - set(item)
        if unknown or missing:
            raise SpecError(
                f"item {i} ({op}): unknown {sorted(unknown)}, missing {sorted(missing)}"
            )
        p = {"op": op, **optional, **{k: v for k, v in item.items() if k != "op"}}
        if "ch" in p:
            p["ch"] = _channels(p["ch"], f"item {i} ({op}) ch", op in VIRTUAL_OPS)
        if op == "rel_logstd":
            p["ref_ch"] = _channels(p["ref_ch"], f"item {i} ref_ch")
            if p["ch"] and not p["ref_ch"]:
                raise SpecError(f"item {i}: rel_logstd needs a non-empty ref_ch")
        if op in ("acf", "dacf"):
            hi = LENGTH - 2 if op == "acf" else LENGTH - 3
            if not isinstance(p["lags"], list | tuple):
                raise SpecError(f"item {i}: lags must be a list")
            p["lags"] = [_lag(v, f"item {i} lags", hi) for v in p["lags"]]
        if op in ("trev", "acfsq"):
            p["lag"] = _lag(p["lag"], f"item {i} lag", LENGTH - 2)
        if op == "c3":  # signed lags: c3(k) = mean over the overlap of z_t^2 z_{t+k}
            if not isinstance(p["lags"], list | tuple):
                raise SpecError(f"item {i}: lags must be a list")
            lags = []
            for v in p["lags"]:
                if not isinstance(v, int) or isinstance(v, bool) or not 1 <= abs(v) <= LENGTH - 2:
                    raise SpecError(
                        f"item {i} lags: nonzero ints with |k| <= {LENGTH - 2}, got {v!r}"
                    )
                lags.append(int(v))
            p["lags"] = lags
        if (
            op == "m3"
        ):  # pairs (j, k), 0 < j < k: m3(j, k) = mean over the overlap of z_t z_{t+j} z_{t+k}
            if not isinstance(p["pairs"], list | tuple):
                raise SpecError(f"item {i}: pairs must be a list")
            pairs = []
            for pr in p["pairs"]:
                ok = isinstance(pr, list | tuple) and len(pr) == 2
                ok = ok and all(isinstance(v, int) and not isinstance(v, bool) for v in pr)
                if not ok or not 0 < pr[0] < pr[1] <= LENGTH - 2:
                    raise SpecError(
                        f"item {i} pairs: [j, k] ints with 0 < j < k <= {LENGTH - 2}, got {pr!r}"
                    )
                pairs.append([int(pr[0]), int(pr[1])])
            p["pairs"] = pairs
        if op == "envacf":  # lags of the ACF of the envelope (LENGTH - 4 values)
            if not isinstance(p["lags"], list | tuple):
                raise SpecError(f"item {i}: lags must be a list")
            p["lags"] = [_lag(v, f"item {i} lags", LENGTH - 5) for v in p["lags"]]
        if op in ("corr", "dcorr"):
            pairs = []
            for pr in p["pairs"]:
                if not isinstance(pr, list | tuple) or len(pr) != 2:
                    raise SpecError(f"item {i}: a pair is two different channels, got {pr!r}")
                i0, i1 = _channels(list(pr), f"item {i} pair", op in VIRTUAL_OPS)
                if i0 == i1:
                    raise SpecError(f"item {i}: a pair is two different channels, got {pr!r}")
                pairs.append([i0, i1])
            p["pairs"] = pairs
        if op == "minirocket":
            d = _lag(p["dilation"], f"item {i} dilation", MAX_DILATION)
            cols = []
            for col in p["cols"]:
                if not isinstance(col, list | tuple) or len(col) != 3:
                    raise SpecError(f"item {i}: a column is [kernel, padded, bias], got {col!r}")
                k, padded, bias = col
                if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < len(MR_INDICES):
                    raise SpecError(f"item {i}: kernel index must be in 0..83, got {k!r}")
                if not isinstance(padded, bool):
                    raise SpecError(f"item {i}: padded must be true/false, got {padded!r}")
                if not np.isfinite(float(bias)):
                    raise SpecError(f"item {i}: bias must be finite, got {bias!r}")
                cols.append([int(k), bool(padded), float(bias)])
            p["dilation"], p["cols"] = d, cols
        if op == "conv_relu_mean":
            d = _lag(p["dilation"], f"item {i} dilation", MAX_DILATION)
            stride = _lag(p["stride"], f"item {i} stride", 8)
            taps = p["taps"]
            if not isinstance(taps, list | tuple) or len(taps) != KLEN:
                raise SpecError(f"item {i}: taps must be {KLEN} numbers, got {taps!r}")
            try:
                taps = [float(w) for w in taps]
                bias = float(p["bias"])
            except (TypeError, ValueError) as exc:
                raise SpecError(f"item {i}: taps and bias must be numbers") from exc
            if not all(np.isfinite(taps)) or not np.isfinite(bias):
                raise SpecError(f"item {i}: taps and bias must be finite")
            p["dilation"], p["stride"], p["taps"], p["bias"] = d, stride, taps, float(p["bias"])
        out.append(p)
    return out


def spec_hash(spec) -> str:
    """sha256 of the canonical JSON of the normalised specification."""
    canon = json.dumps(normalize_spec(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def _n_outputs_of(p: dict) -> int:
    op = p["op"]
    if op in ("rel_logstd", "zcr", "ddstd", "trev", "acfsq"):
        return len(p["ch"])
    if op == "volwin":
        return (len(VOLWIN_W) + 1) * len(p["ch"])
    if op == "env":
        return 2 * len(p["ch"])
    if op == "moments":
        return 2 * len(p["ch"])
    if op == "diffmom":
        return 3 * len(p["ch"])
    if op in ("acf", "dacf", "c3", "envacf"):
        return len(p["ch"]) * len(p["lags"])
    if op in ("corr", "dcorr"):
        return len(p["pairs"])
    if op == "m3":
        return len(p["ch"]) * len(p["pairs"])
    if op == "minirocket":
        return len(p["ch"]) * len(p["cols"])
    if op == "conv_relu_mean":
        return len(p["ch"])
    if op == "sample_rate":
        return 1
    return 0  # fill, znorm: implicit


def n_outputs(spec) -> int:
    return sum(_n_outputs_of(p) for p in normalize_spec(spec))


def _virtual_used(spec: list[dict]) -> list[str]:
    """Virtual channels read by some op with outputs (normalised spec)."""
    used = set()
    for p in spec:
        if _n_outputs_of(p):
            used.update(c for c in p.get("ch", []) if isinstance(c, str))
            used.update(c for pr in p.get("pairs", []) for c in pr if isinstance(c, str))
    return [v for v in VIRTUAL if v in used]


def buf_rows(spec) -> int:
    """Rows of the ``buf`` scratch of ``spec``'s kernel: the 4 channels + one per virtual
    channel up to the last one used (4 when none is used: the delivered layout)."""
    used = _virtual_used(normalize_spec(spec))
    return N_CH + (max(VIRTUAL.index(v) for v in used) + 1 if used else 0)


def v1_spec() -> list[dict]:
    """The 77 features of ``tsclf.features.v1`` in the same order (parity reference)."""
    from tsclf.features.v0 import ACF_LAGS

    spec: list[dict] = []
    for c in range(N_CH):
        spec += [
            {"op": "rel_logstd", "ch": [c], "ref_ch": [0, 1, 2, 3]},
            {"op": "moments", "ch": [c]},
            {"op": "acf", "ch": [c], "lags": [int(v) for v in ACF_LAGS]},
            {"op": "zcr", "ch": [c]},
            {"op": "ddstd", "ch": [c]},
        ]
    for c in (0, 1):
        spec += [
            {"op": "diffmom", "ch": [c]},
            {"op": "trev", "ch": [c], "lag": 3},
            {"op": "acfsq", "ch": [c], "lag": 8},
        ]
    spec += [{"op": "corr", "pairs": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]}]
    spec += [{"op": "sample_rate"}]
    return spec


# ----------------------------------------------------------------------------- cleaning


@njit(**NJIT_OPTS)
def fill_channel(series, c, buf):
    """``tsclf.features.v1.fill_fast`` for channel ``c`` only (same rule, same arithmetic)."""
    L = series.shape[1]
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
        d = abs(v - med)
        n_out += 0.0 if (d <= thr and d < np.inf) else 1.0
    if n_out == 0.0:
        return
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
        if t == L:
            v0 = buf[c, last] if last >= 0 else 0.0
            for u in range(start, L):
                buf[c, u] = v0
        elif last == -1:
            v = buf[c, t]
            for u in range(t):
                buf[c, u] = v
        else:
            v0 = buf[c, last]
            v = buf[c, t]
            for u in range(last + 1, t):
                buf[c, u] = v0 + (v - v0) * (u - last) / (t - last)


@njit(**NJIT_OPTS)
def _repair_channel(buf, c, med, thr):
    """Interpolate the bad points of ``buf[c]`` exactly as the tail of ``fill_channel``."""
    L = LENGTH
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
        if t == L:
            v0 = buf[c, last] if last >= 0 else 0.0
            for u in range(start, L):
                buf[c, u] = v0
        elif last == -1:
            v = buf[c, t]
            for u in range(t):
                buf[c, u] = v
        else:
            v0 = buf[c, last]
            v = buf[c, t]
            for u in range(last + 1, t):
                buf[c, u] = v0 + (v - v0) * (u - last) / (t - last)


@njit(**NJIT_OPTS)
def fill_znorm_channel(series, c, buf):
    """``fill_channel`` + the z-normalisation of channel ``c``; returns ``log(sd + 1e-6)``.

    Compact variant (experiment 20261111-cold-latency): the same rule and arithmetic as
    ``fill_channel`` followed by the generated ``znorm`` block, with branch-free masks (``&``
    instead of ``and``: the statistics loops vectorise, ~2x faster) and the channel sum taken
    in the copy pass when nothing is repaired (one pass less). The sums may round differently
    (reassociation), so values match the reference to ~1e-15, not bit for bit.
    """
    L = LENGTH
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
            ok = np.isfinite(v) & (abs(v - mu) <= lim)
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
    s = 0.0
    for t in range(L):
        v = np.float64(series[c, t])
        buf[c, t] = v
        s += v
        d = abs(v - med)  # NaN for NaN, inf for +-inf: both fail the test below
        n_out += 0.0 if ((d <= thr) & (d < np.inf)) else 1.0
    if n_out != 0.0:
        _repair_channel(buf, c, med, thr)
        s = 0.0
        for t in range(L):
            s += buf[c, t]
    mean = s / L
    s2 = 0.0
    for t in range(L):
        d = buf[c, t] - mean
        s2 += d * d
    sd = np.sqrt(s2 / L)
    inv = 1.0 / sd if sd > 1e-9 else 0.0
    for t in range(L):
        buf[c, t] = (buf[c, t] - mean) * inv
    return np.log(sd + 1e-6)


@njit(**NJIT_OPTS)
def _repair_marked(buf, c):
    """``_repair_channel``'s segments when the bad points of ``buf[c]`` are exactly its NaN
    entries (``fill_znorm_channel_fr`` writes NaN for them; every good point is finite).

    A block of 8 whose sum is not NaN holds no bad point and is skipped in one step (experiment
    20261112-fast-repair: 5.4 bad points of 256 per repaired channel on val_shifted, so the
    point-by-point walk of ``_repair_channel`` was mostly spent on clean points). The finite
    float32 inputs cannot overflow a float64 sum of 8 into inf or NaN.
    """
    L = LENGTH
    last = -1
    t = 0
    while t < L:
        if (t & 7) == 0 and t + 8 <= L:
            s = 0.0
            for u in range(8):
                s += buf[c, t + u]
            if s == s:
                last = t + 7
                t += 8
                continue
        if buf[c, t] == buf[c, t]:
            last = t
            t += 1
            continue
        start = t
        while t < L and buf[c, t] != buf[c, t]:
            t += 1
        if t == L:
            v0 = buf[c, last] if last >= 0 else 0.0
            for u in range(start, L):
                buf[c, u] = v0
        elif last == -1:
            v = buf[c, t]
            for u in range(t):
                buf[c, u] = v
        else:
            v0 = buf[c, last]
            v = buf[c, t]
            for u in range(last + 1, t):
                buf[c, u] = v0 + (v - v0) * (u - last) / (t - last)


@njit(**NJIT_OPTS)
def fill_znorm_channel_fr(series, c, buf):
    """``fill_znorm_channel`` with a cheaper repair (kernel option ``fast_repair``).

    The copy pass writes NaN instead of each bad point: its value is never read, because the
    repair overwrites every bad point (interpolation between good neighbours, or the nearest good
    value at the edges). ``_repair_marked`` then skips the clean blocks. Same rule, same
    arithmetic, same summation loops: bit-identical to ``fill_znorm_channel``
    (``tests/test_fast_repair.py``). Experiment 20261112-fast-repair.
    """
    L = LENGTH
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
            ok = np.isfinite(v) & (abs(v - mu) <= lim)
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
    s = 0.0
    for t in range(L):
        v = np.float64(series[c, t])
        s += v
        d = abs(v - med)  # NaN for NaN, inf for +-inf: both fail the test below
        good = (d <= thr) & (d < np.inf)
        buf[c, t] = v if good else np.nan
        n_out += 0.0 if good else 1.0
    if n_out != 0.0:
        _repair_marked(buf, c)
        s = 0.0
        for t in range(L):
            s += buf[c, t]
    mean = s / L
    s2 = 0.0
    for t in range(L):
        d = buf[c, t] - mean
        s2 += d * d
    sd = np.sqrt(s2 / L)
    inv = 1.0 / sd if sd > 1e-9 else 0.0
    for t in range(L):
        buf[c, t] = (buf[c, t] - mean) * inv
    return np.log(sd + 1e-6)


# ----------------------------------------------------------------------------- code generation


def _rows_of(p: dict) -> dict:
    """Copy of a normalised op with virtual channel names replaced by their ``buf`` rows."""

    def row(c):
        return N_CH + VIRTUAL.index(c) if isinstance(c, str) else c

    q = dict(p)
    if "ch" in q:
        q["ch"] = [row(c) for c in q["ch"]]
    if "pairs" in q:
        q["pairs"] = [[row(a), row(b)] for a, b in q["pairs"]]
    return q


class _Gen:
    """Builds the body of ``features(series, sample_rate, buf, dz, out)``."""

    def __init__(
        self,
        spec: list[dict],
        const_length: bool = True,
        fast_fill: bool = False,
        fuse_lags: bool = False,
        chan_loops: bool = False,
        fuse_more: bool = False,
        fast_repair: bool = False,
    ):
        self.virtual = _virtual_used(spec)  # names; rows N_CH + VIRTUAL.index(name)
        self.chan_loops = chan_loops
        self.fuse_more = fuse_more
        self.fast_repair = fast_repair
        self.need: dict = {}
        self.spec = [_rows_of(p) for p in spec]
        self.const_length = const_length
        self.fast_fill = fast_fill
        self.fuse_lags = fuse_lags
        self.corr01 = None  # accumulator name of sum z0 z1 when the s01 pass computes it
        self.env_row = None  # channel whose ``env`` R is still valid in dz[N_CH] (for envacf)
        self.lines: list[str] = []
        self.done: set = set()

    def emit(self, *lines: str, indent: int = 1) -> None:
        self.lines += ["    " * indent + ln for ln in lines]

    # --- per-channel prerequisites ---------------------------------------------------------
    def channels(self) -> list[int]:
        """Physical channels some op with outputs reads (an op with no outputs costs nothing),
        plus those a used virtual channel is built from."""
        used: set[int] = set()
        for p in self.spec:
            if not _n_outputs_of(p):
                continue
            used.update(p.get("ch", []))
            used.update(p.get("ref_ch", []))
            for pr in p.get("pairs", []):
                used.update(pr)
        if "s01" in self.virtual:
            used.update((0, 1))
        return sorted(c for c in used if c < N_CH)

    def s01(self) -> None:
        """Virtual channel s01 = standardised z0 + z1 into ``buf[N_CH + 0]`` (one pass that
        stores u = z0 + z1 and accumulates its moments, one that scales it). Its variance is
        2 + 2 rho01 for non-constant channels, but it is computed from u itself so that constant
        channels (z = 0) still give unit variance. If the spec has ``corr`` (0, 1), its sum is
        accumulated in the same pass (``self.corr01``)."""
        r = N_CH + VIRTUAL.index("s01")
        fuse = any(
            sorted(pr) == [0, 1] for p in self.spec if p["op"] == "corr" for pr in p["pairs"]
        )
        self.emit(
            "su = 0.0",
            "su2 = 0.0",
            *(["c01 = 0.0"] if fuse else []),
            "for t in range(L):",
            "    a = buf[0, t]",
            "    b = buf[1, t]",
            "    u = a + b",
            f"    buf[{r}, t] = u",
            "    su += u",
            "    su2 += u * u",
            *(["    c01 += a * b"] if fuse else []),
            "m = su / L",
            "var = su2 / L - m * m",
            "sd = np.sqrt(var) if var > 0.0 else 0.0",
            "inv = 1.0 / sd if sd > TINY else 0.0",
            "for t in range(L):",
            f"    buf[{r}, t] = (buf[{r}, t] - m) * inv",
        )
        self.corr01 = "c01" if fuse else None

    def volwin(self, c: int, o: int) -> None:
        """CV of the window means of z^2 for w = 8, 16, 32 (non-overlapping windows) and
        log((max + 1e-3) / (min + 1e-3)) of the w = 32 means -> out[o : o + 4]. The w = 8 means
        go to the scratch row dz[N_CH]; w = 16 and 32 are the means of adjacent pairs."""
        w0 = VOLWIN_W[0]
        n = LENGTH // w0
        self.emit(
            f"zr = dz[{N_CH}]",
            f"for j in range({n}):",
            "    a = 0.0",
            f"    for t in range({w0}):",
            f"        v = buf[{c}, {w0} * j + t]",
            "        a += v * v",
            f"    zr[j] = a / {w0}",
        )
        off = 0
        for k, _ in enumerate(VOLWIN_W):
            if k:  # pairs of the previous level
                prev, n = off, n // 2
                off = prev + 2 * n
                self.emit(
                    f"for j in range({n}):",
                    f"    zr[{off} + j] = (zr[{prev} + 2 * j] + zr[{prev} + 2 * j + 1]) / 2.0",
                )
            self.emit(
                "a = 0.0",
                f"for j in range({n}):",
                f"    a += zr[{off} + j]",
                f"m = a / {n}",
                "a = 0.0",
                f"for j in range({n}):",
                f"    d = zr[{off} + j] - m",
                "    a += d * d",
                f"out[{o + k}] = np.sqrt(a / {n}) / (m if m > TINY else TINY)",
            )
        self.emit(
            f"lo = zr[{off}]",
            "hi = lo",
            f"for j in range(1, {n}):",
            f"    v = zr[{off} + j]",
            "    lo = min(lo, v)",
            "    hi = max(hi, v)",
            f"out[{o + len(VOLWIN_W)}] = np.log((hi + 1e-3) / (lo + 1e-3))",
        )

    def env(self, c: int, o: int) -> None:
        """Phase-plane envelope of MA3(z): m = 3-point moving average (valid, L - 2 values);
        u = m[1:-1] and v = m[2:] - m[:-2] (L - 4 values), each standardised (sd floored at
        TINY); R = u^2 + v^2. out[o] = mean(sqrt R) / sqrt(mean R), out[o + 1] = sd(R) / mean R
        (mean R floored at TINY). m and then R live in the scratch row dz[N_CH]."""
        n = LENGTH - 4
        self.envelope_r(c)
        self.emit(
            "a = 0.0",
            f"for t in range({n}):",
            "    d = zr[t] - mr",
            "    a += d * d",
            "den = mr if mr > TINY else TINY",
            f"out[{o}] = (sq / {n}) / np.sqrt(den)",
            f"out[{o + 1}] = np.sqrt(a / {n}) / den",
        )
        self.env_row = c

    def envelope_r(self, c: int) -> None:
        """R (L - 4 values) of ``env`` into dz[N_CH][:L - 4], its mean in ``mr`` and the sum of
        sqrt R in ``sq``."""
        n = LENGTH - 4
        self.emit(
            f"zr = dz[{N_CH}]",
            "for t in range(L - 2):",
            f"    zr[t] = (buf[{c}, t] + buf[{c}, t + 1] + buf[{c}, t + 2]) / 3.0",
            "su = 0.0",
            "sv = 0.0",
            "for t in range(1, L - 3):",
            "    su += zr[t]",
            "    sv += zr[t + 1] - zr[t - 1]",
            f"mu = su / {n}",
            f"mv = sv / {n}",
            "su = 0.0",
            "sv = 0.0",
            "for t in range(1, L - 3):",
            "    a = zr[t] - mu",
            "    b = zr[t + 1] - zr[t - 1] - mv",
            "    su += a * a",
            "    sv += b * b",
            f"iu = np.sqrt(su / {n})",
            f"iv = np.sqrt(sv / {n})",
            "iu = 1.0 / (iu if iu > TINY else TINY)",
            "iv = 1.0 / (iv if iv > TINY else TINY)",
            "sr = 0.0",
            "sq = 0.0",
            "for t in range(1, L - 3):",
            "    a = (zr[t] - mu) * iu",
            "    b = (zr[t + 1] - zr[t - 1] - mv) * iv",
            "    e = a * a + b * b",
            "    zr[t - 1] = e",  # m[t - 1] is not read again from here on
            "    sr += e",
            "    sq += np.sqrt(e)",
            f"mr = sr / {n}",
        )

    def envacf(self, c: int, lags: list, o: int) -> None:
        """ACF of the envelope R of ``env`` (centred by its mean, normalised by the sum of squares,
        floored at TINY) at ``lags`` -> out[o : o + len(lags)]. Reuses R when ``env`` of the same
        channel was emitted just before with nothing writing dz[N_CH] in between; centres R in
        place (so R is invalid afterwards)."""
        n = LENGTH - 4
        if self.env_row != c:
            self.envelope_r(c)
        self.emit(
            f"zr = dz[{N_CH}]",
            "a = 0.0",
            f"for t in range({n}):",
            "    d = zr[t] - mr",
            "    zr[t] = d",
            "    a += d * d",
            "den = a if a > TINY else TINY",
        )
        if self.fuse_more and len(lags) >= 2:  # one loop over the zero-padded centred R
            self.emit(
                f"for t in range({n}, {n + max(lags)}):",
                "    zr[t] = 0.0",
                *[f"q{j} = 0.0" for j in range(len(lags))],
                f"for t in range({n - min(lags)}):",
                "    u = zr[t]",
                *[f"    q{j} += u * zr[t + {k}]" for j, k in enumerate(lags)],
                *[f"out[{o + j}] = q{j} / den" for j in range(len(lags))],
            )
            self.env_row = None
            return
        for j, k in enumerate(lags):
            self.emit(
                "a = 0.0",
                f"for t in range({n - k}):",
                f"    a += zr[t] * zr[t + {k}]",
                f"out[{o + j}] = a / den",
            )
        self.env_row = None

    def c3(self, c: int, lags: list, o: int) -> None:
        """Third-order cumulants mean(z_t^2 z_{t+k}) over the overlap (divisor L - |k|)."""
        for j, k in enumerate(lags):
            lo, hi, sh = (0, f"L - {k}", f"t + {k}") if k > 0 else (-k, "L", f"t - {-k}")
            self.emit(
                "a = 0.0",
                f"for t in range({lo}, {hi}):",
                f"    u = buf[{c}, t]",
                f"    a += u * u * buf[{c}, {sh}]",
                f"out[{o + j}] = a / (L - {abs(k)})",
            )

    def m3(self, c: int, pairs: list, o: int) -> None:
        """Third-order moments mean(z_t z_{t+j} z_{t+k}), 0 < j < k, over the overlap (divisor
        L - k): the off-diagonal part of the bispectrum that ``c3`` (j = 0 or j = k) misses."""
        for i, (j, k) in enumerate(pairs):
            self.emit(
                "a = 0.0",
                f"for t in range(L - {k}):",
                f"    a += buf[{c}, t] * buf[{c}, t + {j}] * buf[{c}, t + {k}]",
                f"out[{o + i}] = a / (L - {k})",
            )

    def znorm(self, c: int) -> None:
        self.emit(
            f"fill_channel(series, {c}, buf)",
            "s = 0.0",
            "for t in range(L):",
            f"    s += buf[{c}, t]",
            "m = s / L",
            "s2 = 0.0",
            "for t in range(L):",
            f"    d = buf[{c}, t] - m",
            "    s2 += d * d",
            "sd = np.sqrt(s2 / L)",
            f"ls{c} = np.log(sd + 1e-6)",
            "inv = 1.0 / sd if sd > 1e-9 else 0.0",
            "for t in range(L):",
            f"    buf[{c}, t] = (buf[{c}, t] - m) * inv",
        )

    def fill_znorm_loop(self, chans: list[int]) -> None:
        """Cleaning + z-norm of ``chans`` in ONE runtime loop over the channels: one compiled
        copy of ``fill_znorm_channel`` instead of one inlined/specialised copy per channel
        (literal channel arguments made numba compile ``fill_channel`` 4 times, ~9 KB of
        machine code; experiment 20261111-cold-latency)."""
        if not chans:
            return
        self.emit(*[f"ls{c} = 0.0" for c in chans])
        fn = "fill_znorm_channel_fr" if self.fast_repair else "fill_znorm_channel"
        body = [f"lsv = {fn}(series, c, buf)"]
        for i, c in enumerate(chans):
            body += [f"{'if' if i == 0 else 'elif'} c == {c}:", f"    ls{c} = lsv"]
        if chans == list(range(len(chans))):
            self.emit(f"for c in range({len(chans)}):", *["    " + b for b in body])
        else:
            cond = " or ".join(f"c == {c}" for c in chans)
            self.emit(
                f"for c in range({N_CH}):", f"    if {cond}:", *["        " + b for b in body]
            )

    def moments_pass(self, c: int, need: set[str]) -> None:
        """One pass for skew / kurt / m4 / zcr / ddstd, accumulating only what is needed."""
        if (c, "mpass") in self.done:
            return
        if self.chan_loops:
            group = [
                g
                for g in self.mp_channels
                if self.need.get(g) == need and (g, "mpass") not in self.done
            ]
            if len(group) >= 2 and c in group:
                self.moments_loop(group, need)
                return
        self.done.add((c, "mpass"))
        s3, s4 = "s3" in need, "s4" in need
        zc, dd = "zc" in need, "dd" in need
        acc = [f"{v}{c} = 0.0" for v, on in (("s3_", s3), ("s4_", s4), ("dd_", dd)) if on]
        acc += [f"zc_{c} = 0"] if zc else []
        self.emit(*acc)
        body = [f"    z = buf[{c}, t]"]
        if s3 or s4:
            body.append("    z2 = z * z")
        if s3:
            body.append(f"    s3_{c} += z2 * z")
        if s4:
            body.append(f"    s4_{c} += z2 * z2")
        if zc or dd:
            body.append("    if t > 0:")
            if dd:
                body += [f"        d = z - buf[{c}, t - 1]", f"        dd_{c} += d * d"]
            if zc:
                body += [
                    f"        if (z > 0.0) != (buf[{c}, t - 1] > 0.0):",
                    f"            zc_{c} += 1",
                ]
        self.emit("for t in range(L):", *body)

    def moments_loop(self, group: list[int], need: set[str]) -> None:
        """``moments_pass`` of several channels in ONE runtime loop over the channels (option
        ``chan_loops``): one copy of the code instead of one per channel, same arithmetic per
        channel. The per-channel results land in the usual ``s3_{c}``... variables."""
        for g in group:
            self.done.add((g, "mpass"))
        names = [v for v in ("s3", "s4", "dd") if v in need]
        zc = "zc" in need
        self.emit(*[f"{v}_{g} = 0.0" for g in group for v in names])
        self.emit(*([f"zc_{g} = 0" for g in group] if zc else []))
        body = [f"x{v} = 0.0" for v in names] + (["xzc = 0"] if zc else [])
        inner = ["    z = buf[ch_, t]"]
        if "s3" in need or "s4" in need:
            inner.append("    z2 = z * z")
        if "s3" in need:
            inner.append("    xs3 += z2 * z")
        if "s4" in need:
            inner.append("    xs4 += z2 * z2")
        if zc or "dd" in need:
            inner.append("    if t > 0:")
            if "dd" in need:
                inner += ["        d = z - buf[ch_, t - 1]", "        xdd += d * d"]
            if zc:
                inner += [
                    "        if (z > 0.0) != (buf[ch_, t - 1] > 0.0):",
                    "            xzc += 1",
                ]
        body += ["for t in range(L):", *inner]
        body += self._scatter(group, [*names, *(["zc"] if zc else [])], "{v}_{g} = x{v}")
        self.emit(*self._chan_loop(group, body))

    def _chan_loop(self, group: list[int], body: list[str]) -> list[str]:
        """``for ci_ in range(len(group))`` with ``ch_`` = the buf row of iteration ci_."""
        if group == list(range(group[0], group[0] + len(group))):
            ch = f"ci_ + {group[0]}" if group[0] else "ci_"
        else:
            ch = f"({', '.join(map(str, group))},)[ci_]"
        n = len(group)
        # trip count opaque to LLVM (series.shape[0] is 4 at run time): with range(n) it fully
        # unrolls the channel loop and the code is duplicated again; min() keeps it safe. A series
        # with fewer than N_CH rows would skip outputs silently: the (4, 256) shape guard of the
        # model.py wrappers (unexpected shapes -> the neutral series) is REQUIRED with chan_loops.
        trip = f"min(series.shape[0] - {N_CH - n}, {n})"
        return [f"for ci_ in range({trip}):", f"    ch_ = {ch}", *["    " + b for b in body]]

    @staticmethod
    def _scatter(group: list[int], names: list[str], fmt: str) -> list[str]:
        """``if ch_ == g: <fmt for every name>`` chain (per-channel results out of a loop)."""
        out = []
        for i, g in enumerate(group):
            out.append(f"{'if' if i == 0 else 'elif'} ch_ == {g}:")
            out += ["    " + fmt.format(v=v, g=g) for v in names]
        return out

    def diffmom_loop(self, group: list[int]) -> None:
        """``diffmom_pass`` of several channels in one runtime channel loop (``chan_loops``)."""
        for g in group:
            self.done.add((g, "diffmom"))
        self.emit(*[f"{v}{g} = 0.0" for g in group for v in ("dsk", "dku", "tr1_")])
        body = [
            "d1 = 0.0",
            "d2 = 0.0",
            "d3 = 0.0",
            "d4 = 0.0",
            "for t in range(1, L):",
            "    d = buf[ch_, t] - buf[ch_, t - 1]",
            "    e = d * d",
            "    d1 += d",
            "    d2 += e",
            "    d3 += e * d",
            "    d4 += e * e",
            "d1 /= L - 1",
            "d2 /= L - 1",
            "d3 /= L - 1",
            "d4 /= L - 1",
            "var = d2 - d1 * d1",
            "xdsk = 0.0",
            "xdku = 0.0",
            "if var > TINY:",
            "    c3 = d3 - 3.0 * d1 * d2 + 2.0 * d1 * d1 * d1",
            "    c4 = d4 - 4.0 * d1 * d3 + 6.0 * d1 * d1 * d2 - 3.0 * d1 * d1 * d1 * d1",
            "    xdsk = c3 / (var * np.sqrt(var))",
            "    xdku = c4 / (var * var) - 3.0",
            "xtr1_ = d3 / (d2 * np.sqrt(d2)) if d2 > TINY else 0.0",
        ]
        body += self._scatter(group, ["dsk", "dku", "tr1_"], "{v}{g} = x{v}")
        self.emit(*self._chan_loop(group, body))

    def trev_loop(self, group: list[int], k: int, outs: list[int]) -> None:
        """``trev`` at lag ``k`` of several channels in one channel loop -> out[outs[i]]."""
        if outs == list(range(outs[0], outs[0] + len(outs))):
            oi = f"{outs[0]} + ci_"
        else:
            oi = f"({', '.join(map(str, outs))},)[ci_]"
        body = [
            "g2 = 0.0",
            "g3 = 0.0",
            f"for t in range({k}, L):",
            f"    d = buf[ch_, t] - buf[ch_, t - {k}]",
            "    e = d * d",
            "    g2 += e",
            "    g3 += e * d",
            f"g2 /= L - {k}",
            f"g3 /= L - {k}",
            f"out[{oi}] = g3 / (g2 * np.sqrt(g2)) if g2 > TINY else 0.0",
        ]
        self.emit(*self._chan_loop(group, body))

    def trev_loop_one(self, c: int, k: int, o: int) -> None:
        """``trev`` of one channel (same code as the default path)."""
        self.emit(
            "g2 = 0.0",
            "g3 = 0.0",
            f"for t in range({k}, L):",
            f"    d = buf[{c}, t] - buf[{c}, t - {k}]",
            "    e = d * d",
            "    g2 += e",
            "    g3 += e * d",
            f"g2 /= L - {k}",
            f"g3 /= L - {k}",
            f"out[{o}] = g3 / (g2 * np.sqrt(g2)) if g2 > TINY else 0.0",
        )

    def fused_trev(self, c: int, members: list) -> None:
        """Every ``trev`` lag of channel ``c`` in one loop over t >= max lag, plus short loops
        for the first terms of the smaller lags (``fuse_more``). Same terms; the sums may round
        differently."""
        if (c, "trev") in self.done:
            return
        self.done.add((c, "trev"))
        kmax = max(k for k, _ in members)

        def acc(j, k):
            return [
                f"    d = buf[{c}, t] - buf[{c}, t - {k}]",
                "    e = d * d",
                f"    h2_{j} += e",
                f"    h3_{j} += e * d",
            ]

        self.emit(*[f"h{m}_{j} = 0.0" for j in range(len(members)) for m in (2, 3)])
        body = []
        for j, (k, _) in enumerate(members):
            body += acc(j, k)
        self.emit(f"for t in range({kmax}, L):", *body)
        for j, (k, _) in enumerate(members):
            if k < kmax:
                self.emit(f"for t in range({k}, {kmax}):", *acc(j, k))
        for j, (k, oi) in enumerate(members):
            self.emit(
                f"g2 = h2_{j} / (L - {k})",
                f"g3 = h3_{j} / (L - {k})",
                f"out[{oi}] = g3 / (g2 * np.sqrt(g2)) if g2 > TINY else 0.0",
            )

    def fused_c3(self, c: int, members: list, m3: list = ()) -> None:
        """Every ``c3`` lag (and every ``m3`` pair, ``m3`` = [((j, k), out index), ...]) of
        channel ``c`` in one loop (``fuse_more``): z is copied into the zero-padded scratch row
        dz[N_CH] (P zeros before, Q after), so the sum over the overlap is a plain sum over
        t < L. Writes dz[N_CH] (an env R kept there becomes invalid)."""
        if (c, "c3") in self.done:
            return
        self.done.add((c, "c3"))
        P = max([-k for k, _ in members if k < 0] + [0])
        Q = max([k for k, _ in members if k > 0] + [k for (_, k), _ in m3] + [0])
        self.emit(
            f"zr = dz[{N_CH}]",
            *([f"for t in range({P}):", "    zr[t] = 0.0"] if P else []),
            "for t in range(L):",
            f"    zr[t + {P}] = buf[{c}, t]",
            *([f"for t in range(L + {P}, L + {P + Q}):", "    zr[t] = 0.0"] if Q else []),
            *[f"q{j} = 0.0" for j in range(len(members))],
            *[f"w3_{i} = 0.0" for i in range(len(m3))],
            "for t in range(L):",
            f"    u = zr[t + {P}]",
            *(["    u2 = u * u"] if members else []),
            *[f"    q{j} += u2 * zr[t + {P + k}]" for j, (k, _) in enumerate(members)],
            # one product z_t z_{t+j} per distinct j, shared by the pairs (j, k) (amendment 1)
            *[f"    p3_{j} = u * zr[t + {P + j}]" for j in sorted({j for (j, _), _ in m3})],
            *[f"    w3_{i} += p3_{j} * zr[t + {P + k}]" for i, ((j, k), _) in enumerate(m3)],
            *[f"out[{oi}] = q{j} / (L - {abs(k)})" for j, (k, oi) in enumerate(members)],
            *[f"out[{oi}] = w3_{i} / (L - {k})" for i, ((_, k), oi) in enumerate(m3)],
        )
        self.env_row = None

    def diffmom_pass(self, c: int) -> None:
        if (c, "diffmom") in self.done:
            return
        if self.chan_loops:
            group = [g for g in self.dm_channels if (g, "diffmom") not in self.done]
            if len(group) >= 2 and c in group:
                self.diffmom_loop(group)
                return
        self.done.add((c, "diffmom"))
        self.emit(
            "d1 = 0.0",
            "d2 = 0.0",
            "d3 = 0.0",
            "d4 = 0.0",
            "for t in range(1, L):",
            f"    d = buf[{c}, t] - buf[{c}, t - 1]",
            "    e = d * d",
            "    d1 += d",
            "    d2 += e",
            "    d3 += e * d",
            "    d4 += e * e",
            "d1 /= L - 1",
            "d2 /= L - 1",
            "d3 /= L - 1",
            "d4 /= L - 1",
            "var = d2 - d1 * d1",
            f"dsk{c} = 0.0",
            f"dku{c} = 0.0",
            "if var > TINY:",
            "    c3 = d3 - 3.0 * d1 * d2 + 2.0 * d1 * d1 * d1",
            "    c4 = d4 - 4.0 * d1 * d3 + 6.0 * d1 * d1 * d2 - 3.0 * d1 * d1 * d1 * d1",
            f"    dsk{c} = c3 / (var * np.sqrt(var))",
            f"    dku{c} = c4 / (var * var) - 3.0",
            f"tr1_{c} = d3 / (d2 * np.sqrt(d2)) if d2 > TINY else 0.0",
        )

    def dz(self, c: int) -> None:
        """Standardised first differences of z into dz[c, :L-1]."""
        if (c, "dz") in self.done:
            return
        self.done.add((c, "dz"))
        self.emit(
            "s = 0.0",
            "for t in range(1, L):",
            f"    d = buf[{c}, t] - buf[{c}, t - 1]",
            f"    dz[{c}, t - 1] = d",
            "    s += d",
            "m = s / (L - 1)",
            "s2 = 0.0",
            "for t in range(L - 1):",
            f"    d = dz[{c}, t] - m",
            "    s2 += d * d",
            "sd = np.sqrt(s2 / (L - 1))",
            "inv = 1.0 / sd if sd > 1e-9 else 0.0",
            "for t in range(L - 1):",
            f"    dz[{c}, t] = (dz[{c}, t] - m) * inv",
        )

    def minirocket(self, c: int, d: int, cols: list, base: int) -> None:
        """All columns of one (channel, dilation) group; PPV to out[base + j].

        z[c] is copied into the zero-padded row ``dz[N_CH]`` (4d zeros on each side), so every
        position, edges included, uses the same branch-free formula: sm = 9-tap sum once per
        position, then 3 taps and a comparison per column. Interior positions count for every
        column; the 4d edge positions only for padded ones. Generated code grows linearly with
        the number of columns (unrolled per-tap bounds checks made LLVM superlinear).
        """
        P = 4 * d
        self.emit(
            f"zr = dz[{N_CH}]",  # 1-D view: 2-D indexing in the hot loop costs ~2x (measured)
            f"for t in range({P}):",
            "    zr[t] = 0.0",
            f"    zr[L + {P} + t] = 0.0",
            "for t in range(L):",
            f"    zr[t + {P}] = buf[{c}, t]",
        )
        taps = " + ".join(f"zr[t + {k * d}]" for k in range(KLEN))

        def counts(members):
            out = []
            for j, (k, _, bias) in members:
                three = " + ".join(f"zr[t + {i * d}]" for i in MR_INDICES[k])
                out.append(f"    n{j} += 1 if 3.0 * ({three}) - sm > {bias!r} else 0")
            return out

        everyone = list(enumerate(cols))
        padded = [(j, col) for j, col in everyone if col[1]]
        self.emit(*[f"n{j} = 0" for j, _ in everyone])
        self.emit(f"for t in range({P}, L - {P}):", f"    sm = {taps}", *counts(everyone))
        if padded:
            for lo, hi in (("0", f"{P}"), (f"L - {P}", "L")):
                self.emit(f"for t in range({lo}, {hi}):", f"    sm = {taps}", *counts(padded))
        for j, (_, pad, _) in everyone:
            self.emit(
                f"out[{base + j}] = n{j} / L" if pad else f"out[{base + j}] = n{j} / (L - {2 * P})"
            )

    def _lag_groups(self) -> dict:
        """{(op, c): [(lag, output index), ...]} over ALL ``acf`` / ``acfsq`` ops of the spec."""
        groups: dict = {}
        o = 0
        for p in self.spec:
            if p["op"] == "acf":
                for c in p["ch"]:
                    for lag in p["lags"]:
                        groups.setdefault(("acf", c), []).append((lag, o))
                        o += 1
            elif p["op"] == "acfsq":
                for c in p["ch"]:
                    groups.setdefault(("acfsq", c), []).append((p["lag"], o))
                    o += 1
            else:
                o += _n_outputs_of(p)
        return groups

    def fused_lags(self, op: str, c: int, members: list) -> None:
        """Every lag of ``op`` (acf or acfsq) on channel ``c`` in ONE loop (experiment
        20261111-cold-latency): the series (z, or z^2 - 1 for acfsq) is copied into the
        zero-padded row ``dz[N_CH]``, so all lags share one branch-free loop over t < L - min
        lag (the padded zeros add exact zeros). Same terms as the per-lag loops; the sums may
        round differently (reassociation)."""
        if (op, c, "fused") in self.done:
            return
        if self.chan_loops and self._fused_lags_loop(op, c, members):
            return
        self.done.add((op, c, "fused"))
        lags = [lag for lag, _ in members]
        src = f"buf[{c}, t]" if op == "acf" else f"buf[{c}, t] * buf[{c}, t] - 1.0"
        self.emit(
            f"zr = dz[{N_CH}]",
            "for t in range(L):",
            f"    zr[t] = {src}",
            f"for t in range(L, L + {max(lags)}):",
            "    zr[t] = 0.0",
            *[f"q{j} = 0.0" for j in range(len(lags))],
            f"for t in range(L - {min(lags)}):",
            "    u = zr[t]",
            *[f"    q{j} += u * zr[t + {lag}]" for j, lag in enumerate(lags)],
        )
        if op == "acf":
            self.emit(*[f"out[{oi}] = q{j} / L" for j, (_, oi) in enumerate(members)])
        else:
            self.emit(
                f"vq = s4_{c} / L - 1.0",
                *[
                    f"out[{oi}] = q{j} / (L * vq) if vq > TINY else 0.0"
                    for j, (_, oi) in enumerate(members)
                ],
            )

    def _fused_lags_loop(self, op: str, c: int, members: list) -> bool:
        """``fused_lags`` of every channel with the same lags as ``c`` in one channel loop
        (``chan_loops``) when their output positions are affine in the channel index."""
        lags = [lag for lag, _ in members]
        group = sorted(
            g
            for (o2, g), m in self.lag_groups.items()
            if o2 == op and [lag for lag, _ in m] == lags and (op, g, "fused") not in self.done
        )
        if c not in group or len(group) < 2:
            return False
        outs = [[oi for _, oi in self.lag_groups[(op, g)]] for g in group]
        stride = outs[1][0] - outs[0][0]
        if any(
            outs[i][j] != outs[0][j] + i * stride
            for i in range(len(group))
            for j in range(len(lags))
        ):
            return False
        for g in group:
            self.done.add((op, g, "fused"))
            if op == "acfsq":
                self.moments_pass(g, self.need[g])
        src = "buf[ch_, t]" if op == "acf" else "buf[ch_, t] * buf[ch_, t] - 1.0"
        body = [
            f"zr = dz[{N_CH}]",
            "for t in range(L):",
            f"    zr[t] = {src}",
            f"for t in range(L, L + {max(lags)}):",
            "    zr[t] = 0.0",
            *[f"q{j} = 0.0" for j in range(len(lags))],
            f"for t in range(L - {min(lags)}):",
            "    u = zr[t]",
            *[f"    q{j} += u * zr[t + {lag}]" for j, lag in enumerate(lags)],
        ]
        if op == "acf":
            body += [f"out[{oi} + {stride} * ci_] = q{j} / L" for j, oi in enumerate(outs[0])]
        else:
            body += [f"s4c = s4_{group[0]}"]
            body += [ln for g in group[1:] for ln in (f"if ch_ == {g}:", f"    s4c = s4_{g}")]
            body += ["vq = s4c / L - 1.0"]
            body += [
                f"out[{oi} + {stride} * ci_] = q{j} / (L * vq) if vq > TINY else 0.0"
                for j, oi in enumerate(outs[0])
            ]
        self.emit(*self._chan_loop(group, body))
        return True

    def lag_loop(self, op: str, c: int, lag: int, o: int) -> None:
        """One ``acf`` / ``acfsq`` output with its own loop (same code as the default path)."""
        if op == "acf":
            self.emit(
                "a = 0.0",
                f"for t in range(L - {lag}):",
                f"    a += buf[{c}, t] * buf[{c}, t + {lag}]",
                f"out[{o}] = a / L",
            )
        else:
            self.emit(
                f"vq = s4_{c} / L - 1.0",
                "a = 0.0",
                f"for t in range(L - {lag}):",
                f"    u = buf[{c}, t]",
                f"    v = buf[{c}, t + {lag}]",
                "    a += (u * u - 1.0) * (v * v - 1.0)",
                f"out[{o}] = a / (L * vq) if vq > TINY else 0.0",
            )

    def _conv_groups(self) -> dict:
        """``conv_relu_mean`` ops with the same channels, dilation and stride >= 2, anywhere.

        They are computed in one loop (one pass over z, every filter's accumulator updated per
        position): measured -23 % for 4 filters at stride 2, but slower at stride 1, so stride-1
        filters keep one loop each. Each op still writes to its own output positions. Returns
        {index of the first op of a group: [(op, base), ...], index of the others: "member"}.
        """
        bases, o = [], 0
        for p in self.spec:
            bases.append(o)
            o += _n_outputs_of(p)
        by_key: dict = {}
        for i, p in enumerate(self.spec):
            if p["op"] == "conv_relu_mean" and p["stride"] >= 2 and p["ch"]:
                key = (tuple(p["ch"]), p["dilation"], p["stride"])
                by_key.setdefault(key, []).append(i)
        groups: dict = {}
        for all_idxs in by_key.values():
            for start in range(0, len(all_idxs), MAX_FUSED_FILTERS):  # the measured group size
                idxs = all_idxs[start : start + MAX_FUSED_FILTERS]
                if len(idxs) > 1:
                    groups[idxs[0]] = [(self.spec[k], bases[k]) for k in idxs]
                    for k in idxs[1:]:
                        groups[k] = "member"
        return groups

    def conv_relu_mean(self, members: list) -> None:
        """Filters ``members`` = [(op, base)] (same channels, dilation, stride), one loop per
        channel; op output for the c-th listed channel goes to out[base + c]."""
        d, st, chans = members[0][0]["dilation"], members[0][0]["stride"], members[0][0]["ch"]
        n_pos = (LENGTH - 8 * d - 1) // st + 1  # valid outputs, as torch Conv1d
        for ci, c in enumerate(chans):
            self.emit(*[f"a{m} = 0.0" for m in range(len(members))])
            body = []
            for m, (q, _) in enumerate(members):
                taps = " + ".join(
                    f"{w!r} * buf[{c}, t + {k * d}]" for k, w in enumerate(q["taps"]) if w != 0.0
                )
                body += [
                    f"    v{m} = {q['bias']!r} + {taps or '0.0'}",
                    f"    a{m} += v{m} if v{m} > 0.0 else 0.0",
                ]
            self.emit(f"for t in range(0, L - {8 * d}, {st}):", *body)
            for m, (_, base) in enumerate(members):
                self.emit(f"out[{base + ci}] = a{m} / {n_pos}")

    # --- the whole body --------------------------------------------------------------------
    def build(self) -> str:
        self.emit(
            f"L = {LENGTH}" if self.const_length else "L = series.shape[1]"
        )  # literal: faster
        if self.virtual:  # numba does not bounds-check: a 4-row buf must fail loudly (setup)
            rows = N_CH + 1 + max(VIRTUAL.index(v) for v in self.virtual)
            self.emit(
                f"if buf.shape[0] < {rows}:",
                f"    raise ValueError('this spec needs buf with {rows} rows (buf_rows)')",
            )
        if self.fast_fill:
            self.fill_znorm_loop(self.channels())
        else:
            for c in self.channels():
                self.znorm(c)
        if "s01" in self.virtual:
            self.s01()
        # what each channel's moments pass must accumulate
        need: dict[int, set[str]] = self.need
        for p in self.spec:
            for c in p.get("ch", []):
                n = need.setdefault(c, set())
                n.update(
                    {
                        "moments": {"s3", "s4"},
                        "zcr": {"zc"},
                        "ddstd": {"dd"},
                        "acfsq": {"s4"},
                    }.get(p["op"], set())
                )
        o = 0
        fused = self._conv_groups()  # spec index -> its group (first member only); members skipped
        lag_groups = self._lag_groups() if self.fuse_lags else {}
        self.lag_groups = lag_groups
        aux_free = AUX_SHAPE[1] - LENGTH  # padding available in dz[N_CH]
        moment_ops = ("moments", "zcr", "ddstd", "acfsq")
        self.mp_channels = sorted({c for p in self.spec if p["op"] in moment_ops for c in p["ch"]})
        self.dm_channels = sorted({c for p in self.spec if p["op"] == "diffmom" for c in p["ch"]})
        trev_groups: dict = {}
        c3_groups: dict = {}
        m3_groups: dict = {}
        o2 = 0
        for p in self.spec:
            for ci, c in enumerate(p.get("ch", [])):
                if p["op"] == "trev":
                    trev_groups.setdefault(c, []).append((p["lag"], o2 + ci))
                elif p["op"] == "c3":
                    for j, k in enumerate(p["lags"]):
                        c3_groups.setdefault(c, []).append((k, o2 + ci * len(p["lags"]) + j))
                elif p["op"] == "m3":
                    for j, pr in enumerate(p["pairs"]):
                        m3_groups.setdefault(c, []).append(
                            ((pr[0], pr[1]), o2 + ci * len(p["pairs"]) + j)
                        )
            o2 += _n_outputs_of(p)

        def c3_fusable(c) -> bool:
            """The lags of ``c3`` and ``m3`` on channel ``c`` fit in the padding of dz[N_CH]."""
            ks = [abs(k) for k, _ in c3_groups.get(c, [])]
            ks += [k for (_, k), _ in m3_groups.get(c, [])]
            return 2 * max(ks) <= aux_free

        for idx, p in enumerate(self.spec):
            op = p["op"]
            if not _n_outputs_of(p):
                continue
            if op not in _KEEPS_ENV_ROW and op != "envacf":
                self.env_row = None  # this op may write dz[N_CH] (env sets it again below)
            if op == "rel_logstd":
                ref = " + ".join(f"ls{r}" for r in p["ref_ch"])
                for c in p["ch"]:
                    self.emit(f"out[{o}] = ls{c} - ({ref}) / {len(p['ref_ch'])}")
                    o += 1
            elif op in ("moments", "zcr", "ddstd"):
                for c in p["ch"]:
                    self.moments_pass(c, need[c])
                    if op == "moments":
                        self.emit(f"out[{o}] = s3_{c} / L", f"out[{o + 1}] = s4_{c} / L - 3.0")
                        o += 2
                    elif op == "zcr":
                        self.emit(f"out[{o}] = zc_{c} / (L - 1)")
                        o += 1
                    else:
                        self.emit(f"out[{o}] = np.sqrt(dd_{c} / (L - 1))")
                        o += 1
            elif op in ("acf", "acfsq") and self.fuse_lags:
                lags = p["lags"] if op == "acf" else [p["lag"]]
                for c in p["ch"]:
                    if op == "acfsq":
                        self.moments_pass(c, need[c])
                    members = lag_groups[(op, c)]
                    if max(lag for lag, _ in members) <= aux_free:
                        self.fused_lags(op, c, members)
                    else:  # a lag beyond the padding of dz[N_CH]: the delivered per-lag loops
                        for j, lag in enumerate(lags):
                            self.lag_loop(op, c, lag, o + j)
                    o += len(lags)
            elif op == "acf":
                for c in p["ch"]:
                    for lag in p["lags"]:
                        self.emit(
                            "a = 0.0",
                            f"for t in range(L - {lag}):",
                            f"    a += buf[{c}, t] * buf[{c}, t + {lag}]",
                            f"out[{o}] = a / L",
                        )
                        o += 1
            elif op == "diffmom":
                for c in p["ch"]:
                    self.diffmom_pass(c)
                    self.emit(
                        f"out[{o}] = dsk{c}", f"out[{o + 1}] = dku{c}", f"out[{o + 2}] = tr1_{c}"
                    )
                    o += 3
            elif op == "trev" and (self.chan_loops or self.fuse_more):
                rest = []
                for ci, c in enumerate(p["ch"]):
                    if self.fuse_more and len(trev_groups[c]) >= 2:
                        self.fused_trev(c, trev_groups[c])
                    else:
                        rest.append((c, o + ci))
                if self.chan_loops and len(rest) >= 2:
                    self.trev_loop([c for c, _ in rest], p["lag"], [oi for _, oi in rest])
                else:
                    for c, oi in rest:
                        self.trev_loop_one(c, p["lag"], oi)
                o += len(p["ch"])
            elif op == "trev":
                k = p["lag"]
                for c in p["ch"]:
                    self.emit(
                        "g2 = 0.0",
                        "g3 = 0.0",
                        f"for t in range({k}, L):",
                        f"    d = buf[{c}, t] - buf[{c}, t - {k}]",
                        "    e = d * d",
                        "    g2 += e",
                        "    g3 += e * d",
                        f"g2 /= L - {k}",
                        f"g3 /= L - {k}",
                        f"out[{o}] = g3 / (g2 * np.sqrt(g2)) if g2 > TINY else 0.0",
                    )
                    o += 1
            elif op == "acfsq":
                k = p["lag"]
                for c in p["ch"]:
                    self.moments_pass(c, need[c])
                    self.emit(
                        f"vq = s4_{c} / L - 1.0",
                        "a = 0.0",
                        f"for t in range(L - {k}):",
                        f"    u = buf[{c}, t]",
                        f"    v = buf[{c}, t + {k}]",
                        "    a += (u * u - 1.0) * (v * v - 1.0)",
                        f"out[{o}] = a / (L * vq) if vq > TINY else 0.0",
                    )
                    o += 1
            elif op == "corr":
                for i, j in p["pairs"]:
                    if self.corr01 and sorted((i, j)) == [0, 1]:
                        self.emit(f"out[{o}] = {self.corr01} / L")
                        o += 1
                        continue
                    self.emit(
                        "a = 0.0",
                        "for t in range(L):",
                        f"    a += buf[{i}, t] * buf[{j}, t]",
                        f"out[{o}] = a / L",
                    )
                    o += 1
            elif op == "dcorr":
                for i, j in p["pairs"]:
                    self.dz(i)
                    self.dz(j)
                    self.emit(
                        "a = 0.0",
                        "for t in range(L - 1):",
                        f"    a += dz[{i}, t] * dz[{j}, t]",
                        f"out[{o}] = a / (L - 1)",
                    )
                    o += 1
            elif op == "dacf":
                for c in p["ch"]:
                    self.dz(c)
                    for lag in p["lags"]:
                        self.emit(
                            "a = 0.0",
                            f"for t in range(L - 1 - {lag}):",
                            f"    a += dz[{c}, t] * dz[{c}, t + {lag}]",
                            f"out[{o}] = a / (L - 1)",
                        )
                        o += 1
            elif op == "minirocket":
                for c in p["ch"]:
                    self.minirocket(c, p["dilation"], p["cols"], o)
                    o += len(p["cols"])
            elif op == "conv_relu_mean":
                group = fused.get(idx)
                if group == "member":  # emitted with the first op of its group
                    o += len(p["ch"])
                    continue
                self.conv_relu_mean(group or [(p, o)])
                o += len(p["ch"])
            elif op == "volwin":
                for c in p["ch"]:
                    self.volwin(c, o)
                    o += len(VOLWIN_W) + 1
            elif op == "env":
                for c in p["ch"]:
                    self.env(c, o)
                    o += 2
            elif op == "c3" and self.fuse_more and all(c3_fusable(c) for c in p["ch"]):
                for c in p["ch"]:
                    self.fused_c3(c, c3_groups[c], m3_groups.get(c, []))
                    o += len(p["lags"])
            elif op == "c3":
                for c in p["ch"]:
                    self.c3(c, p["lags"], o)
                    o += len(p["lags"])
            elif op == "m3":
                for c in p["ch"]:
                    # with fuse_more the pairs ride in the channel's c3 loop, emitted once at the
                    # first of the two ops (a c3 op that fell back to plain loops because of
                    # another of its channels just writes the same values again)
                    if self.fuse_more and c3_fusable(c):
                        self.fused_c3(c, c3_groups.get(c, []), m3_groups[c])
                    else:
                        self.m3(c, p["pairs"], o)
                    o += len(p["pairs"])
            elif op == "envacf":
                for c in p["ch"]:
                    self.envacf(c, p["lags"], o)
                    o += len(p["lags"])
            elif op == "sample_rate":
                self.emit(f"out[{o}] = sample_rate")
                o += 1
        self.emit(
            f"for i in range({o}):",
            "    if not np.isfinite(out[i]):",
            "        out[i] = 0.0",
        )
        self.n_out = o
        return "\n".join(self.lines)


def generate_source(
    spec,
    const_length: bool = True,
    fast_fill: bool = False,
    fuse_lags: bool = False,
    *,
    chan_loops: bool = False,
    fuse_more: bool = False,
    fast_repair: bool = False,
) -> tuple[str, int]:
    """Source of ``features(series, sample_rate, buf, dz, out)`` and its number of outputs.

    ``const_length=False`` reads L from the input (the kernel delivered before experiment
    20261111-v2-minirocket-b; kept for latency A/B tests): same values, slower loops.
    ``fast_fill=True`` cleans and z-normalises with ``fill_znorm_channel`` in one channel loop
    (experiment 20261111-cold-latency): same values up to summation rounding, less code.
    ``fuse_lags=True`` computes all ``acf`` (resp. ``acfsq``) lags of a channel in one loop.
    Experiment 20261111-v7-cold (less machine code: the harness times cold calls):
    ``chan_loops=True`` runs the per-channel passes (moments, diffmom, trev, fused acf/acfsq) of
    channels with identical parameters in ONE runtime loop over the channels (one copy of the
    code, same arithmetic per channel); ``fuse_more=True`` computes all ``trev`` / ``c3`` /
    ``envacf`` lags of a channel in one loop each (same terms, sums may round differently).
    ``fast_repair=True`` (needs ``fast_fill``; experiment 20261112-fast-repair) cleans with
    ``fill_znorm_channel_fr``: bit-identical values, cheaper spike / NaN repair.
    """
    if fast_repair and not fast_fill:
        raise ValueError("fast_repair needs fast_fill (it replaces fill_znorm_channel)")
    g = _Gen(
        normalize_spec(spec), const_length, fast_fill, fuse_lags, chan_loops, fuse_more, fast_repair
    )
    body = g.build()
    return "def features(series, sample_rate, buf, dz, out):\n" + body + "\n", g.n_out


_COMPILED: dict[str, object] = {}  # generated source -> compiled dispatcher


def _compile(source: str, name: str):
    """Compile generated ``source`` once per process (same source -> same dispatcher)."""
    if source not in _COMPILED:
        ns = {
            "np": np,
            "fill_channel": fill_channel,
            "fill_znorm_channel": fill_znorm_channel,
            "fill_znorm_channel_fr": fill_znorm_channel_fr,
            "TINY": TINY,
        }
        exec(compile(source, f"<generated {name}>", "exec"), ns)  # spec literals only
        _COMPILED[source] = njit(**NJIT_OPTS)(ns[name])
    return _COMPILED[source]


def make_features(spec, fast_fill: bool = False, fuse_lags: bool = False, **more):
    """``features(series, sample_rate, buf, dz, out)`` for ``spec`` and its number of outputs.

    ``buf``: (4, L) float64 scratch (overwritten with the z-scores); ``dz``: ``AUX_SHAPE`` float64
    scratch; ``out``: (n_outputs,) float64. ``sample_rate`` must already be finite.
    """
    source, n = generate_source(spec, fast_fill=fast_fill, fuse_lags=fuse_lags, **more)
    return _compile(source, "features"), n


def features_batch(
    series: np.ndarray,
    sample_rate: np.ndarray,
    spec,
    fast_fill: bool = False,
    fuse_lags: bool = False,
    **more,
) -> np.ndarray:
    """(n, 4, L) series + (n,) sample rates -> (n, n_outputs) float64 (training helper)."""
    fn, n_out = make_features(spec, fast_fill, fuse_lags, **more)
    series = np.asarray(series)
    X = np.empty((len(series), n_out))
    buf, dz = np.empty((buf_rows(spec), series.shape[2])), np.empty(AUX_SHAPE)
    for j in range(len(series)):
        sr = float(sample_rate[j])
        fn(
            np.ascontiguousarray(series[j]),
            sr if np.isfinite(sr) else NEUTRAL_SAMPLE_RATE,
            buf,
            dz,
            X[j],
        )
    return X


def zscore_batch(series: np.ndarray) -> np.ndarray:
    """(n, 4, L) raw series -> (n, 4, L) cleaned, z-normalised channels exactly as the v2 kernel
    sees them (training helper: fit MiniRocket biases on this, not on median/MAD scores)."""
    fn, _ = make_features([{"op": "corr", "pairs": [[0, 1], [2, 3]]}])
    series = np.asarray(series)
    Z = np.empty(series.shape, dtype=np.float64)
    buf, aux, out = np.empty((N_CH, series.shape[2])), np.empty(AUX_SHAPE), np.empty(2)
    for j in range(len(series)):
        fn(np.ascontiguousarray(series[j]), 1.0, buf, aux, out)
        Z[j] = buf
    return Z


# ----------------------------------------------------------------------------- predictor


def work_size_v2(n_features: int, hidden: int, n_classes: int = 8, rows: int = N_CH) -> int:
    return rows * LENGTH + AUX_SHAPE[0] * AUX_SHAPE[1] + n_features + hidden + n_classes


def make_predictor_v2(
    spec,
    W1,
    b1,
    W2,
    b2,
    float32_weights: bool = True,
    const_length: bool = True,
    fast_fill: bool = False,
    fuse_lags: bool = False,
    w1_int16: bool = False,
    chan_loops: bool = False,
    fuse_more: bool = False,
    fast_repair: bool = False,
    shape_guard: bool = False,
):
    """Kernel ``predict(series, sample_rate, work) -> int`` for ``spec`` + an MLP 1 x H.

    Same structure as ``tsclf.inference.mlp_v1.make_predictor(fast=True)``: generated features,
    weights frozen as constants, output layer by hidden unit skipping inactive ones, one
    preallocated ``work`` array of ``work_size_v2(F, H)`` float64 elements. Raises ``SpecError``
    if ``spec`` cannot be generated or does not match ``W1``. ``fast_fill``, ``fuse_lags``: see
    ``generate_source`` (defaults False = the delivered kernel). ``w1_int16`` stores W1 as int16
    with one float64 scale per hidden unit (a quarter of the float64 bytes, half of float32;
    relative error <= 1/65534 of the column's max weight), applied after the dot product.
    ``shape_guard=True`` (experiment 20261112-lean-predict) returns ABSTAIN (-1) without reading
    ``series`` when its shape is not (N_CH, LENGTH), so the Python caller needs no shape check;
    it is a separate kernel: without it the code is exactly the unguarded one.
    """
    source, n_out = generate_source(
        spec,
        const_length,
        fast_fill,
        fuse_lags,
        chan_loops=chan_loops,
        fuse_more=fuse_more,
        fast_repair=fast_repair,
    )
    wtype = np.float32 if float32_weights else np.float64
    W1 = np.ascontiguousarray(W1, dtype=wtype)
    W2 = np.ascontiguousarray(W2, dtype=wtype)
    b1 = np.ascontiguousarray(b1, dtype=np.float64)
    b2 = np.ascontiguousarray(b2, dtype=np.float64)
    F, H = W1.shape
    K = W2.shape[1]
    if F != n_out or W2.shape[0] != H or b1.shape != (H,) or b2.shape != (K,):
        raise SpecError(
            f"weights do not match the specification: spec has {n_out} features, "
            f"W1 {W1.shape}, b1 {b1.shape}, W2 {W2.shape}, b2 {b2.shape}"
        )
    features = _compile(source, "features")
    rows = buf_rows(spec)  # N_CH unless the spec uses a virtual channel
    n_buf, n_dz = rows * LENGTH, AUX_SHAPE[0] * AUX_SHAPE[1]
    o_feat = n_buf + n_dz
    o_hid = o_feat + F
    o_acc = o_hid + H
    neutral = NEUTRAL_SAMPLE_RATE
    if w1_int16:
        if shape_guard:
            raise SpecError("w1_int16 does not support shape_guard")
        return _predictor_int16(features, W1, b1, W2, b2, rows, o_feat, o_hid, o_acc), work_size_v2(
            F, H, K, rows
        )
    if shape_guard:

        @njit(**NJIT_OPTS)
        def guarded(series, sample_rate, work):
            if series.shape[0] != N_CH or series.shape[1] != LENGTH:
                return -1  # ABSTAIN: a valid answer, and nothing out of bounds is read
            if not np.isfinite(sample_rate):
                sample_rate = neutral
            buf = work[:n_buf].reshape(rows, LENGTH)
            dz = work[n_buf:o_feat].reshape(AUX_SHAPE[0], AUX_SHAPE[1])
            feat = work[o_feat:o_hid]
            hid = work[o_hid:o_acc]
            acc = work[o_acc : o_acc + K]
            features(series, sample_rate, buf, dz, feat)
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

        return guarded, work_size_v2(F, H, K, rows)

    @njit(**NJIT_OPTS)
    def predict(series, sample_rate, work):
        if not np.isfinite(sample_rate):
            sample_rate = neutral
        buf = work[:n_buf].reshape(rows, LENGTH)
        dz = work[n_buf:o_feat].reshape(AUX_SHAPE[0], AUX_SHAPE[1])
        feat = work[o_feat:o_hid]
        hid = work[o_hid:o_acc]
        acc = work[o_acc : o_acc + K]
        features(series, sample_rate, buf, dz, feat)
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

    return predict, work_size_v2(F, H, K, rows)


def quantize_int16(W):
    """(F, H) weights -> (int16 Q, float64 scale per column) with W ~= Q * scale."""
    W = np.asarray(W, dtype=np.float64)
    amax = np.abs(W).max(axis=0)
    scale = np.where(amax > 0, amax / 32767.0, 1.0)
    Q = np.clip(np.rint(W / scale), -32767, 32767).astype(np.int16)
    return np.ascontiguousarray(Q), np.ascontiguousarray(scale)


def _predictor_int16(features, W1, b1, W2, b2, rows, o_feat, o_hid, o_acc):
    """``make_predictor_v2`` kernel with W1 stored as int16 (``w1_int16=True``)."""
    Q1, s1 = quantize_int16(W1)
    F, H = Q1.shape
    K = W2.shape[1]
    n_buf = rows * LENGTH
    neutral = NEUTRAL_SAMPLE_RATE

    @njit(**NJIT_OPTS)
    def predict(series, sample_rate, work):
        if not np.isfinite(sample_rate):
            sample_rate = neutral
        buf = work[:n_buf].reshape(rows, LENGTH)
        dz = work[n_buf:o_feat].reshape(AUX_SHAPE[0], AUX_SHAPE[1])
        feat = work[o_feat:o_hid]
        hid = work[o_hid:o_acc]
        acc = work[o_acc : o_acc + K]
        features(series, sample_rate, buf, dz, feat)
        for j in range(H):
            hid[j] = 0.0
        for i in range(F):
            f = feat[i]
            for j in range(H):
                hid[j] += f * Q1[i, j]
        for j in range(H):
            hid[j] = b1[j] + hid[j] * s1[j]
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

    return predict
