"""Bank simulator: five banks, each owning ONLY its own partition.

Simulation disclaimer: all five banks run in one process on one computer.
This demonstrates the federated protocol, not a deployment across real bank
servers (no TLS, auth, or secure aggregation in this prototype).
"""
from __future__ import annotations

import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from src.local_training import train_local
from src.evaluation import evaluate_params

BANK_LABEL = {0: "Bank A", 1: "Bank B", 2: "Bank C", 3: "Bank D", 4: "Bank E"}


class Bank:
    def __init__(self, bank_id: int, X_train, y_train, X_test, y_test):
        self.id = bank_id
        self.label = BANK_LABEL.get(bank_id, f"Bank {bank_id}")
        self.X_train, self.y_train = np.asarray(X_train, float), np.asarray(y_train, int)
        self.X_test, self.y_test = np.asarray(X_test, float), np.asarray(y_test, int)

    @property
    def n_train(self): return len(self.X_train)

    def local_baseline(self, seed=C.RANDOM_SEED, iters=C.LOCAL_MAX_ITER):
        """Local-only model from zero init (never sees other banks)."""
        n = self.X_train.shape[1]
        zero = {"coef": np.zeros((1, n)), "intercept": np.zeros(1)}
        upd = train_local(self.X_train, self.y_train, init_params=zero,
                          local_iters=iters, seed=seed + self.id)
        params = {"coef": upd["coef"], "intercept": upd["intercept"]}
        return params, evaluate_params(params, self.X_test, self.y_test)

    def train_round(self, global_params, local_iters, seed, clip_norm, noise_sigma, rng):
        from src.update_protection import clip_update, add_noise
        upd = train_local(self.X_train, self.y_train, init_params=global_params,
                          local_iters=local_iters, seed=seed + self.id)
        protected, clip_info = clip_update(upd, global_params, clip_norm)
        protected = add_noise(protected, clip_norm, noise_sigma, rng)
        info = {"bank_id": self.id, "n": upd["n"], "warning": upd.get("warning"),
                **clip_info, "noise_sigma": float(noise_sigma or 0.0)}
        return protected, info
