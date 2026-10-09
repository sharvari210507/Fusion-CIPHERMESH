"""FedAvg implemented from scratch (spec section 7).

w_global = sum_k (n_k / sum_j n_j) * w_k      (applied to coef_ and intercept_)

The coordinator NEVER receives raw rows — only {"coef","intercept","n","meta"}.
Every update is validated (shape, finiteness, schema tag); bad updates are
rejected and recorded in the audit ledger instead of silently averaged.
"""
from __future__ import annotations

import numpy as np


def _as_vec(d: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(d["coef"], dtype=float), np.asarray(d["intercept"], dtype=float)


def validate_update(upd: dict, ref_coef_shape: tuple, ref_int_shape: tuple) -> tuple[bool, str]:
    """Check one bank update. Returns (accepted, reason)."""
    try:
        if not isinstance(upd, dict) or any(k not in upd for k in ("coef", "intercept", "n", "meta")):
            return False, "missing keys (need coef/intercept/n/meta)"
        c, b = _as_vec(upd)
        if c.shape != tuple(ref_coef_shape):
            return False, f"coef shape {c.shape} != expected {tuple(ref_coef_shape)}"
        if b.shape != tuple(ref_int_shape):
            return False, f"intercept shape {b.shape} != expected {tuple(ref_int_shape)}"
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(b))):
            return False, "non-finite values (NaN/inf)"
        n = int(upd["n"])
        if n <= 0:
            return False, f"sample count n={n} invalid"
        return True, "ok"
    except Exception as e:
        return False, f"malformed: {e}"


def fedavg(updates: list[dict], ref_coef_shape: tuple, ref_int_shape: tuple):
    """Sample-count-weighted average of ACCEPTED updates.

    Returns (global_params {"coef","intercept"}, accepted_idx, rejected [(idx, reason)]).
    Raises ValueError if nothing was accepted.
    """
    accepted, rejected, weights = [], [], []
    total = 0
    for i, u in enumerate(updates):
        ok, reason = validate_update(u, ref_coef_shape, ref_int_shape)
        if ok:
            accepted.append(i)
            total += int(u["n"])
        else:
            rejected.append((i, reason))
    if not accepted:
        raise ValueError(f"All updates rejected: {rejected}")
    coef = np.zeros(ref_coef_shape, dtype=float)
    inter = np.zeros(ref_int_shape, dtype=float)
    for i in accepted:
        w = int(updates[i]["n"]) / total
        c, b = _as_vec(updates[i])
        coef += w * c
        inter += w * b
        weights.append({"idx": i, "weight": w, "n": int(updates[i]["n"])})
    return {"coef": coef, "intercept": inter}, accepted, rejected, weights
