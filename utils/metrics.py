"""Separation metrics (SDR / SIR / SAR) via mir_eval."""
from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from mir_eval.separation import bss_eval_sources
except Exception:  # pragma: no cover - mir_eval optional at import time
    bss_eval_sources = None


def _bss(reference: np.ndarray, estimate: np.ndarray):
    if bss_eval_sources is None:
        raise ImportError("mir_eval is required for metric computation")
    return bss_eval_sources(reference, estimate, compute_permutation=True)


def compute_sdr(reference: np.ndarray, estimate: np.ndarray) -> Dict[str, float]:
    """reference / estimate: [num_sources, samples]. Returns mean SDR/SIR/SAR.

    NOTE: these are mixture-level means over the sources. For a per-instrument
    breakdown use ``compute_sdr_per_source`` -- attributing this single mean to
    each source's category separately is wrong (see [FIX #8]).
    """
    sdr, sir, sar, _ = _bss(reference, estimate)
    return {
        "sdr": float(np.mean(sdr)),
        "sir": float(np.mean(sir)),
        "sar": float(np.mean(sar)),
    }


def compute_sdr_per_source(reference: np.ndarray, estimate: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-source SDR/SIR/SAR, aligned with the REFERENCE ordering.

    [FIX #8] The per-instrument table used to append the mixture-level mean SDR
    to every category present in the mixture, so a violin mixed with a tuba
    credited both categories with the same number. That makes the per-instrument
    breakdown meaningless -- it can only ever reproduce the overall mean with
    different sample counts.

    ``mir_eval.separation.bss_eval_sources(..., compute_permutation=True)``
    internally evaluates every estimate/reference pairing, selects the best
    permutation ``popt``, and returns ``sdr[popt, arange(nsrc)]``. The returned
    arrays are therefore indexed by REFERENCE index, so element ``i`` is the
    score for reference source ``i`` -- which is what lets us attribute it to
    that source's own instrument category.

    Returns arrays of length num_sources plus the chosen permutation.
    """
    sdr, sir, sar, perm = _bss(reference, estimate)
    return {
        "sdr": np.asarray(sdr, dtype=float),
        "sir": np.asarray(sir, dtype=float),
        "sar": np.asarray(sar, dtype=float),
        "perm": np.asarray(perm),
    }
