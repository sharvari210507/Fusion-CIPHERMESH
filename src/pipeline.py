"""End-to-end experiment runner with saved, reproducible artifacts."""
from __future__ import annotations

import csv
import json
import os, sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from src.bank_simulator import Bank
from src.coordinator import Coordinator
from src.data_loader import bank_partitions, class_counts, load_raw
from src.preprocessing import Preprocessor, feature_names


def run_experiment(subset_total=C.SUBSET_TOTAL, seed=C.RANDOM_SEED,
                   n_rounds=C.N_ROUNDS_DEFAULT, local_iters=C.LOCAL_MAX_ITER,
                   bank_ids=None, clip_norm=C.CLIP_NORM_DEFAULT,
                   noise_sigma=C.NOISE_MULT_DEFAULT, threshold=C.DECISION_THRESHOLD,
                   force_synthetic=False, save=True, out_dir=C.RESULTS_DIR,
                   models_dir=C.MODELS_DIR):
    bank_ids = sorted(bank_ids if bank_ids is not None else C.BANK_IDS)
    ts = datetime.now(timezone.utc)
    df, meta = load_raw(subset_total, seed, force_synthetic)
    parts = bank_partitions(df, C.TEST_SIZE, seed)
    # Fit preprocessor on combined TRAIN only
    import pandas as pd
    train_df = pd.concat([parts[b]["train"] for b in bank_ids])
    pre = Preprocessor().fit(train_df)
    banks = []
    for b in bank_ids:
        banks.append(Bank(
            b, pre.transform(parts[b]["train"]), parts[b]["train"][C.TARGET_COL].values,
            pre.transform(parts[b]["test"]), parts[b]["test"][C.TARGET_COL].values))
    # Local-only baselines
    baselines = {}
    for bk in banks:
        params, m = bk.local_baseline(seed, local_iters)
        m["threshold"] = threshold
        from src.evaluation import evaluate_params
        m = evaluate_params(params, bk.X_test, bk.y_test, threshold)
        baselines[bk.id] = {"metrics": m,
                            "coef_norm": float(np.linalg.norm(params["coef"]))}
    coord = Coordinator(banks, pre, seed)
    coord.run(n_rounds, bank_ids=bank_ids, local_iters=local_iters,
              clip_norm=clip_norm, noise_sigma=noise_sigma)
    res = coord.results()
    # divergence table: local vs federated per bank (last round global)
    divergence = {}
    for b in bank_ids:
        fed = res["history"][-1]["per_bank"][b] if res["history"] else None
        loc = baselines[b]["metrics"]
        divergence[b] = {"local": loc, "federated": fed,
                         "delta_f1": _sub(fed, loc, "f1"),
                         "delta_pr_auc": _sub(fed, loc, "pr_auc"),
                         "delta_recall": _sub(fed, loc, "recall")}
    out = {
        "experiment": "CIPHERMESH",
        "generated_at": ts.isoformat(),
        "config": {"dataset": meta, "seed": seed, "subset_total": subset_total,
                   "banks": bank_ids, "n_rounds": n_rounds, "local_iters": local_iters,
                   "clip_norm": clip_norm, "noise_sigma": noise_sigma,
                   "threshold": threshold, "features": feature_names(),
                   "model": C.MODEL_PARAMS, "force_synthetic": force_synthetic},
        "class_counts": class_counts(parts),
        "warnings": {str(b): parts[b]["warnings"] for b in bank_ids},
        "baselines": {str(k): v for k, v in baselines.items()},
        "divergence": {str(k): v for k, v in divergence.items()},
        "history": res["history"],
        "audit": res["audit"],
        "ledger": coord.ledger.events,
        "ledger_verify": coord.ledger.verify(),
        "limitations": [
            "Banks simulated on one computer — not a real deployment.",
            "Synthetic/HF-synthetic data; not real customer transactions.",
            "Model updates can leak information; zero raw rows != zero risk.",
            "Clipping/noise here are NOT formal differential privacy.",
            "Hash ledger is tamper-evident, not proof of honest updates.",
        ],
    }
    if save:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        stamp = ts.strftime("%Y%m%d_%H%M%S")
        with open(f"{out_dir}/experiment_{stamp}.json", "w") as f:
            json.dump(_jsonable(out), f, indent=2)
        # CSV: per-bank divergence
        with open(f"{out_dir}/divergence_{stamp}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bank", "local_f1", "fed_f1", "d_f1", "local_pr_auc",
                        "fed_pr_auc", "d_pr_auc", "local_recall", "fed_recall",
                        "d_recall", "n_fraud_test"])
            for b in bank_ids:
                d = divergence[b]
                w.writerow([b, d["local"]["f1"], (d["federated"] or {}).get("f1"),
                            d["delta_f1"], d["local"]["pr_auc"],
                            (d["federated"] or {}).get("pr_auc"), d["delta_pr_auc"],
                            d["local"]["recall"], (d["federated"] or {}).get("recall"),
                            d["delta_recall"], d["local"]["n_fraud"]])
        np.savez(f"{models_dir}/global_{stamp}.npz",
                 coef=res["global_params"]["coef"],
                 intercept=res["global_params"]["intercept"])
        pre.save(f"{models_dir}/preprocessor_{stamp}.pkl")
        out["_saved"] = {"stamp": stamp}
    return out, coord, pre


def _sub(a, b, k):
    try:
        if a is None or a.get(k) is None or b.get(k) is None:
            return None
        return float(a[k] - b[k])
    except Exception:
        return None


def _jsonable(o):
    import numpy as _np
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    return o
