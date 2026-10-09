"""Federated coordinator: distributes globals, collects updates, FedAvgs, evaluates.

Coordinator inputs per bank per round: {"coef","intercept","n","meta"} (+ clip
diagnostics). Raw transaction rows are NEVER an input — enforced by signature
(run_round takes Bank objects and reads only .X_train via the bank method;
the aggregation function receives parameter dicts only) and by tests.
"""
from __future__ import annotations

import copy
import os, sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from src.bank_simulator import Bank
from src.evaluation import evaluate_params
from src.fedavg import fedavg
from src.preprocessing import N_FEATURES
from src.privacy_audit import PrivacyAudit
from src.trust_ledger import TrustLedger

REF_COEF = (1, N_FEATURES)
REF_INT = (1,)


class Coordinator:
    def __init__(self, banks: list[Bank], preprocessor, seed=C.RANDOM_SEED):
        self.banks = {b.id: b for b in banks}
        self.pre = preprocessor
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.global_params = {"coef": np.zeros(REF_COEF), "intercept": np.zeros(REF_INT)}
        self.audit = PrivacyAudit()
        self.ledger = TrustLedger()
        self.history: list[dict] = []
        self.ledger.append("experiment_start", "coordinator", 0,
                           {"banks": sorted(self.banks), "seed": seed})

    # -- one round ----------------------------------------------------------
    def run_round(self, round_no: int, bank_ids: list[int] | None = None,
                  local_iters: int = C.LOCAL_MAX_ITER,
                  clip_norm: float | None = C.CLIP_NORM_DEFAULT,
                  noise_sigma: float = C.NOISE_MULT_DEFAULT):
        bank_ids = sorted(bank_ids if bank_ids is not None else self.banks.keys())
        self.ledger.append("global_distributed", "coordinator", round_no,
                           {"banks": bank_ids})
        updates, infos = [], []
        for bid in bank_ids:
            b = self.banks[bid]
            self.ledger.append("local_train_start", b.label, round_no, {"n": b.n_train})
            self.audit.log("local_train_start", b.label, f"n={b.n_train}", round_no)
            upd, info = b.train_round(self.global_params, local_iters,
                                      self.seed + round_no * 100, clip_norm,
                                      noise_sigma, self.rng)
            self.ledger.append("local_train_end", b.label, round_no,
                               {"norm_after": round(info["norm_after"], 4),
                                "clipped": info["clipped"]})
            updates.append(upd)
            infos.append(info)
        self.ledger.append("updates_validated", "coordinator", round_no,
                           {"submitted": len(updates)})
        new_global, accepted, rejected, weights = fedavg(updates, REF_COEF, REF_INT)
        for j, i in enumerate(accepted):
            self.audit.record_update(True, f"Bank {bank_ids[i]}", round_no)
        for i, reason in rejected:
            self.audit.record_update(False, f"Bank {bank_ids[i]}", round_no)
            self.ledger.append("update_rejected", "coordinator", round_no,
                               {"bank": bank_ids[i], "reason": reason})
        self.global_params = new_global
        self.ledger.append("aggregated", "coordinator", round_no,
                           {"accepted": len(accepted), "rejected": len(rejected),
                            "weights": [{**w} for w in weights],
                            "clip_norm": clip_norm, "noise_sigma": noise_sigma})
        self.audit.finish_round([f"Bank {b}" for b in bank_ids], round_no)
        # per-bank evaluation of the NEW global
        per_bank = {}
        for bid in bank_ids:
            b = self.banks[bid]
            m = evaluate_params(self.global_params, b.X_test, b.y_test)
            per_bank[bid] = m
            self.ledger.append("bank_evaluated", f"Bank {bid}", round_no,
                               {"f1": m["f1"], "pr_auc": m["pr_auc"],
                                "n_fraud": m["n_fraud"]})
        rec = {"round": round_no, "banks": bank_ids, "weights": weights,
               "accepted": accepted, "rejected": rejected,
               "clip_info": infos, "per_bank": per_bank,
               "global_norm": float(np.linalg.norm(new_global["coef"]))}
        self.history.append(rec)
        return rec

    def run(self, n_rounds=C.N_ROUNDS_DEFAULT, **kw):
        for r in range(1, n_rounds + 1):
            self.run_round(r, **kw)
        return self.results()

    def results(self):
        return {"global_params": copy.deepcopy(self.global_params),
                "history": copy.deepcopy(self.history),
                "audit": self.audit.summary(),
                "events": copy.deepcopy(self.audit.events),
                "ledger_ok": self.ledger.verify(),
                "generated_at": datetime.now(timezone.utc).isoformat()}
