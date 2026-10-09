"""Update protection: clipping (+ optional Gaussian noise) experiment.

Clipping bounds how far one bank can pull the global model in a round:
  delta = w_local - w_global; if ||delta|| > C: delta *= C/||delta||.
Noise (optional): delta += N(0, (sigma*C)^2). This is a DP-style mechanism
but NOT a formal differential-privacy guarantee (no accounting) — the UI and
README say so explicitly. Clipping alone is not a privacy guarantee either.
"""
from __future__ import annotations

import numpy as np


def delta_norm(local: dict, glob: dict) -> float:
    dc = np.asarray(local["coef"], float) - np.asarray(glob["coef"], float)
    db = np.asarray(local["intercept"], float) - np.asarray(glob["intercept"], float)
    return float(np.sqrt(np.sum(dc ** 2) + np.sum(db ** 2)))


def clip_update(local: dict, glob: dict, clip_norm: float | None):
    """Return (protected_update, info{norm_before, norm_after, clipped})."""
    if clip_norm is None:
        return {"coef": np.array(local["coef"], float),
                "intercept": np.array(local["intercept"], float),
                "n": local["n"], "meta": local.get("meta", {})}, \
            {"norm_before": delta_norm(local, glob), "norm_after": delta_norm(local, glob),
             "clipped": False, "clip_norm": None}
    before = delta_norm(local, glob)
    dc = np.asarray(local["coef"], float) - np.asarray(glob["coef"], float)
    db = np.asarray(local["intercept"], float) - np.asarray(glob["intercept"], float)
    clipped = False
    if before > clip_norm and before > 0:
        s = clip_norm / before
        dc, db = dc * s, db * s
        clipped = True
    out = {"coef": np.asarray(glob["coef"], float) + dc,
           "intercept": np.asarray(glob["intercept"], float) + db,
           "n": local["n"], "meta": {**local.get("meta", {}), "clipped": clipped}}
    after = float(np.sqrt(np.sum(dc ** 2) + np.sum(db ** 2)))
    return out, {"norm_before": before, "norm_after": after,
                 "clipped": clipped, "clip_norm": float(clip_norm)}


def add_noise(upd: dict, clip_norm: float | None, sigma: float,
              rng: np.random.Generator) -> dict:
    """Gaussian noise on a (clipped) update. sigma=0 -> unchanged."""
    if sigma is None or sigma <= 0:
        return {**upd, "meta": {**upd.get("meta", {}), "noise_sigma": 0.0}}
    scale = float(sigma * (clip_norm if clip_norm else 1.0))
    out = {"coef": np.asarray(upd["coef"], float) + rng.normal(0, scale, np.asarray(upd["coef"]).shape),
           "intercept": np.asarray(upd["intercept"], float) + rng.normal(0, scale, np.asarray(upd["intercept"]).shape),
           "n": upd["n"], "meta": {**upd.get("meta", {}), "noise_sigma": float(sigma),
                                   "noise_scale": scale}}
    return out
