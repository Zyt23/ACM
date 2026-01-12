# -*- coding: utf-8 -*-
"""
Mechanism-inspired anomaly synthesis for ACM windows.

Window shape: [L, 6] where 6 columns correspond to:
  [BYPASS_V, DISCH_T, RAM_I_DR, RAM_O_DR, FLOW, COMPR_T]
(works for PACK1/PACK2 because you already align feature order that way)
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple, Optional
import numpy as np


class AnomType(IntEnum):
    NORMAL = 0
    ENERGY_RATIO = 1
    MISALIGN_TEMP = 2
    MISALIGN_VALVE = 3
    DRIFT = 4
    OSCILLATION = 5
    STUCK_VALVE = 6
    NOISE = 7


@dataclass(frozen=True)
class ACMColIndex:
    BYPASS: int = 0
    DISCH: int = 1
    RAM_I: int = 2
    RAM_O: int = 3
    FLOW: int = 4
    COMPR: int = 5


def _shift_1d(x: np.ndarray, k: int) -> np.ndarray:
    """Edge-padded shift. k>0 means delay (shift right)."""
    if k == 0:
        return x.copy()
    L = x.shape[0]
    out = np.empty_like(x)
    if k > 0:
        out[:k] = x[0]
        out[k:] = x[:L - k]
    else:
        kk = -k
        out[L - kk:] = x[-1]
        out[:L - kk] = x[kk:]
    return out


def _shift_window(win: np.ndarray, cols: list[int], k: int) -> np.ndarray:
    out = win.copy()
    for c in cols:
        out[:, c] = _shift_1d(out[:, c], k)
    return out


def _scale_delta(x: np.ndarray, scale: float, ref: str = "median") -> np.ndarray:
    if ref == "first":
        r = float(x[0])
    else:
        r = float(np.nanmedian(x))
    d = x - r
    return r + d * float(scale)


def _add_drift(x: np.ndarray, k: float) -> np.ndarray:
    L = x.shape[0]
    t = np.linspace(0.0, 1.0, L, dtype=np.float32)
    return x + k * t


def _add_oscillation(x: np.ndarray, amp: float, cycles: float) -> np.ndarray:
    L = x.shape[0]
    t = np.arange(L, dtype=np.float32)
    # cycles in window, e.g. 0.5 ~ 2.0
    w = 2.0 * np.pi * float(cycles) / max(1.0, float(L))
    return x + float(amp) * np.sin(w * t)


def apply_anomaly(
    win: np.ndarray,
    kind: AnomType,
    rng: np.random.Generator,
    idx: ACMColIndex = ACMColIndex(),
    params: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Args:
      win: [L,6] float32/float64 (prefer standardized space)
      kind: anomaly type
      rng: numpy generator
    Returns:
      win2, meta(dict)
    """
    assert win.ndim == 2 and win.shape[1] == 6, f"expect [L,6], got {win.shape}"
    L = win.shape[0]
    p = params or {}
    meta = {"kind": int(kind)}

    if kind == AnomType.NORMAL:
        return win.copy(), meta

    out = win.copy()

    # Parameters presets (in standardized unit)
    # You can tune these after looking at normal feature distributions.
    if kind == AnomType.ENERGY_RATIO:
        # distort the relative amplitude between DISCH and COMPR
        scale_choices = p.get("scale_choices", [0.6, 0.8, 1.25, 1.6])
        scale = float(rng.choice(scale_choices))
        # randomly pick to scale DISCH or COMPR
        if bool(rng.integers(0, 2)):
            out[:, idx.DISCH] = _scale_delta(out[:, idx.DISCH], scale=scale, ref="median")
            meta.update({"scale_col": "DISCH", "scale": scale})
        else:
            out[:, idx.COMPR] = _scale_delta(out[:, idx.COMPR], scale=scale, ref="median")
            meta.update({"scale_col": "COMPR", "scale": scale})
        return out, meta

    if kind == AnomType.MISALIGN_TEMP:
        # shift temperature channels relative to valves/flow
        k_choices = p.get("k_choices", [-6, -4, -2, 2, 4, 6])  # steps
        k = int(rng.choice(k_choices))
        out = _shift_window(out, cols=[idx.DISCH, idx.COMPR], k=k)
        meta.update({"shift_cols": "TEMP", "k": k})
        return out, meta

    if kind == AnomType.MISALIGN_VALVE:
        # shift valve/control channels relative to temperature
        k_choices = p.get("k_choices", [-6, -4, -2, 2, 4, 6])
        k = int(rng.choice(k_choices))
        out = _shift_window(out, cols=[idx.BYPASS, idx.RAM_I, idx.RAM_O], k=k)
        meta.update({"shift_cols": "VALVE", "k": k})
        return out, meta

    if kind == AnomType.DRIFT:
        # add slow drift to temperature (simulate degradation)
        drift_k_choices = p.get("drift_k_choices", [-0.6, -0.4, 0.4, 0.6])  # std units per window
        k = float(rng.choice(drift_k_choices))
        which = "COMPR" if bool(rng.integers(0, 2)) else "DISCH"
        if which == "COMPR":
            out[:, idx.COMPR] = _add_drift(out[:, idx.COMPR], k=k)
        else:
            out[:, idx.DISCH] = _add_drift(out[:, idx.DISCH], k=k)
        meta.update({"drift_col": which, "k": k})
        return out, meta

    if kind == AnomType.OSCILLATION:
        amp_choices = p.get("amp_choices", [0.15, 0.25, 0.35])
        cyc_choices = p.get("cyc_choices", [0.5, 1.0, 1.5, 2.0])
        amp = float(rng.choice(amp_choices))
        cyc = float(rng.choice(cyc_choices))
        which = "COMPR" if bool(rng.integers(0, 2)) else "DISCH"
        if which == "COMPR":
            out[:, idx.COMPR] = _add_oscillation(out[:, idx.COMPR], amp=amp, cycles=cyc)
        else:
            out[:, idx.DISCH] = _add_oscillation(out[:, idx.DISCH], amp=amp, cycles=cyc)
        meta.update({"osc_col": which, "amp": amp, "cycles": cyc})
        return out, meta

    if kind == AnomType.STUCK_VALVE:
        # simulate valve stuck: flatten BYPASS or RAMs for a portion
        col = int(rng.choice([idx.BYPASS, idx.RAM_I, idx.RAM_O]))
        a = int(rng.integers(0, max(1, L // 3)))
        b = int(rng.integers(min(L, a + L // 3), L))
        v = float(out[a, col])
        out[a:b, col] = v
        meta.update({"stuck_col": col, "a": a, "b": b})
        return out, meta

    if kind == AnomType.NOISE:
        sigma_choices = p.get("sigma_choices", [0.05, 0.08, 0.12])
        sigma = float(rng.choice(sigma_choices))
        # add noise mainly to temperature
        noise = rng.normal(0.0, sigma, size=(L,)).astype(out.dtype)
        if bool(rng.integers(0, 2)):
            out[:, idx.COMPR] = out[:, idx.COMPR] + noise
            meta.update({"noise_col": "COMPR", "sigma": sigma})
        else:
            out[:, idx.DISCH] = out[:, idx.DISCH] + noise
            meta.update({"noise_col": "DISCH", "sigma": sigma})
        return out, meta

    raise ValueError(f"Unknown anomaly kind: {kind}")
