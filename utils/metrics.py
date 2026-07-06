"""Separation metrics (SDR / SIR / SAR) via mir_eval."""
from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from mir_eval.separation import bss_eval_sources
except Exception:  # pragma: no cover - mir_eval optional at import time
    bss_eval_sources = None


def compute_sdr(reference: np.ndarray, estimate: np.ndarray) -> Dict[str, float]:
    """reference / estimate: [num_sources, samples]. Returns mean SDR/SIR/SAR."""
    if bss_eval_sources is None:
        raise ImportError("mir_eval is required for metric computation")
    sdr, sir, sar, _ = bss_eval_sources(reference, estimate, compute_permutation=True)
    return {
        "sdr": float(np.mean(sdr)),
        "sir": float(np.mean(sir)),
        "sar": float(np.mean(sar)),
    }
