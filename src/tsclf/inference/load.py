"""Build the inference kernel of an artifact (``model.py`` calls this in ``Model.__init__``).

An artifact whose ``config.json["features"]`` is a list (a v2 feature specification) gets the
generated v2 kernel; without it, or with the label ``"v1"``, the v1 kernel. Any problem
(unknown primitive, hash or shape mismatch) raises here, during the untimed setup, never in
``predict``. stdlib + numpy + numba.
"""

from __future__ import annotations

import numpy as np

from tsclf.features.v2 import SpecError, make_predictor_v2, spec_hash
from tsclf.inference.mlp_v1 import make_predictor, work_size


def build_kernel(
    config: dict, weights, const_length: bool = True, **options
) -> tuple[object, np.ndarray]:
    """``(kernel, work)``: ``kernel(series, sample_rate, work) -> int`` and its work array.

    ``const_length=False``: latency A/B reference only (v2 kernel before 20261111-v2-minirocket-b).
    ``options``: v2 kernel variants of experiment 20261111-cold-latency (``fast_fill``,
    ``fuse_lags``; 20261111-v7-cold: ``chan_loops``, ``fuse_more``; see ``generate_source``).
    The artifact's own ``config["kernel_options"]`` are applied first (the delivered artifact
    names the kernel it was timed with); explicit ``options`` override them. Variants change
    latency only, never a decision (parity tests).
    """
    W1, b1, W2, b2 = (np.asarray(weights[k]) for k in ("W1", "b1", "W2", "b2"))
    spec = config.get("features")
    if spec is None or spec == "v1":  # v1 artifact ("features": "v1" is a label, not a spec)
        kernel = make_predictor(W1, b1, W2, b2, float32_weights=True)
        return kernel, np.zeros(work_size(W1.shape[1], W2.shape[1]))
    if not isinstance(spec, list):
        raise SpecError(f"config.json: 'features' must be 'v1' or a v2 specification, got {spec!r}")
    if config.get("features_hash") != spec_hash(spec):
        raise SpecError("config.json: features_hash is missing or does not match the specification")
    options = {**config.get("kernel_options", {}), **options}
    kernel, size = make_predictor_v2(
        spec, W1, b1, W2, b2, float32_weights=True, const_length=const_length, **options
    )
    return kernel, np.zeros(size)
