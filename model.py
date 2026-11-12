"""Contest entrypoint: thin wrapper that loads artifacts/current and delegates to src/tsclf.

Contract (contest/README.md): ``Model()`` loads everything (untimed, <= 120 s);
``predict(x)`` returns a native ``int`` in 0..7 or -1 (abstain) and never raises.
Only stdlib, numpy and numba are imported on this path.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SRC = str(HERE / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tsclf.features.v1 import LENGTH, N_CH  # noqa: E402
from tsclf.inference.load import build_kernel  # noqa: E402
from tsclf.inference.runtime import synthetic_samples  # noqa: E402

ABSTAIN = -1  # same value as competition_api.ABSTAIN; not imported to stay independent of it
ARTIFACT_DIR = HERE / "artifacts" / "current"
SHAPE = (N_CH, LENGTH)
FLOAT32 = np.dtype(np.float32)
NEUTRAL_SAMPLE_RATE = 1.5


class Model:
    def __init__(self) -> None:
        config = json.loads((ARTIFACT_DIR / "config.json").read_text(encoding="utf-8"))
        self.name = config["model"]
        self.fallback = int(config["fallback_class"])
        w = np.load(ARTIFACT_DIR / "weights.npz")
        # one kernel call with 3 arguments, weights frozen as float32 constants, no allocations;
        # v2 artifacts (config "features") get a kernel generated from their specification.
        # Both kernels leave the output layer in the last n_classes slots of work (logits()).
        # shape_guard: ABSTAIN for a wrong shape, without reading it (20261112-lean-predict)
        self._kernel, self._work = build_kernel(config, w, shape_guard=True)
        self._n_classes = int(w["W2"].shape[1])
        self._neutral = np.zeros(SHAPE, dtype=np.float32)  # stands in for unexpected inputs
        self._neutral.setflags(write=False)
        # compile + warm up on normal / partial-NaN / all-NaN inputs (untimed setup), for the
        # read-only arrays the harness sends and for writable ones (another numba signature)
        for x in [*synthetic_samples(), self._neutral]:
            for arr in (x, x.copy()):
                for _ in range(3):
                    self._kernel(arr, 1.0, self._work)
        # unseen types raise TypeError (-> fallback) instead of compiling; absent without numba
        disable = getattr(self._kernel, "disable_compile", None)
        if disable is not None:
            disable()
        gc.collect()
        gc.freeze()  # long-lived objects out of the GC's way during the scored phase

    def predict(self, x) -> int:
        # no Python shape / dtype checks (20261112-lean-predict, Colab A/B x0.910): the kernel
        # guards the shape, and an unexpected type fails at dispatch (compilation is disabled)
        try:
            try:
                sr = float(x.meta["sample_rate"])
            except Exception:
                sr = NEUTRAL_SAMPLE_RATE
            return self._kernel(x.series, sr, self._work)
        except Exception:
            return self.fallback

    def logits(self, x) -> np.ndarray:
        """Class scores (8,) behind ``predict``: never called by the harness (evaluation only).

        The kernel leaves the output layer in the last ``n_classes`` slots of ``work``
        (``tsclf.inference.mlp_v1.work_size``); ``argmax`` of the result is ``predict(x)``.
        """
        self.predict(x)
        return self._work[-self._n_classes :].copy()
