"""E1-E6 experiments using the same engine. Saved to results/experiments.json."""
import json
import numpy as np
import pandas as pd
import joblib
from . import data as D
from . import features as F
from . import metrics as M
from .federated import BankClient, fedavg
from .jobs import _sigmoid, _validate_cfg
from . import database as db
from .config import RESULTS_DIR, MODELS_DIR, EXP_DEFAULT


def _quick_train(banks, cfg, seed, stop=None):
    rng = np.random.RandomState(seed)
    mode = cfg.get("feature_mode", "raw")
    nfeat = len(F.feature_names(mode))
    w = np.zeros(nfeat + 1)
    clients = [BankClient(b, D.bank_partition("train", b), seed) for b in banks]
    c2 = dict(cfg); c2["seed"] = seed
    for r in range(1, cfg.get("rounds", 5) + 1):
        if stop is not None and stop.is_set():
            break
        ups = [c.train_round(w, nfeat, c2, r, rng) for c in clients]
        w = w + fedavg(ups)
    return w


def _eval(w, mode, thr=0.5):
    test = D.load_split("test")
    X = F.transform(test, mode).values; y = test["isFraud"].values
    s = _sigmoid(X @ w[:-1] + w[-1])
    return M.full_metrics(y, s, thr), s, y


def run_suite(jid, cfg, stop):
    seeds = cfg.get("seeds", EXP_DEFAULT["seeds"])
    rounds = cfg.get("rounds", 5); epochs = cfg.get("local_epochs", 1)
    out = {"config": cfg, "E": {}}
    mode = "raw"
    # E1 local only
    e1 = {}
    for b in range(5):
        ms = []
        for s in seeds:
            df = D.bank_partition("train", b)
            c = BankClient(b, df, s)
            rng = np.random.RandomState(s)
            nfeat = len(F.feature_names(mode)); w = np.zeros(nfeat + 1)
            cc = dict(feature_mode=mode, local_epochs=rounds * epochs, clip_norm=None,
                      noise_multiplier=0.0, sampling_frac=1.0, seed=s)
            ups = [c.train_round(w, nfeat, cc, 1, rng)]
            w = w + fedavg(ups)
            m, _, _ = _eval(w, mode)
            ms.append(m["pr_auc"])
        e1[str(b)] = {"mean_pr_auc": float(np.mean(ms)), "std": float(np.std(ms))}
    out["E"]["E1_local"] = e1
    # E2 federation size
    e2 = {}
    for k in [1, 2, 3, 4, 5]:
        ms = []
        for s in seeds:
            rng = np.random.RandomState(s)
            order = list(rng.permutation(5))[:k]
            w = _quick_train(order, dict(feature_mode=mode, rounds=rounds,
                                         local_epochs=epochs, clip_norm=None,
                                         noise_multiplier=0.0, sampling_frac=1.0), s, stop)
            m, _, _ = _eval(w, mode)
            ms.append(m["pr_auc"])
        e2[str(k)] = {"mean_pr_auc": float(np.mean(ms)), "std": float(np.std(ms))}
    out["E"]["E2_fedsize"] = e2
    # E3 pooled reference
    ms = []
    for s in seeds:
        df = D.load_split("train")
        c = BankClient(0, df, s)
        rng = np.random.RandomState(s)
        nfeat = len(F.feature_names(mode)); w = np.zeros(nfeat + 1)
        cc = dict(feature_mode=mode, local_epochs=rounds * epochs, clip_norm=None,
                  noise_multiplier=0.0, sampling_frac=1.0, seed=s)
        ups = [c.train_round(w, nfeat, cc, 1, rng)]
        w = w + fedavg(ups)
        m, _, _ = _eval(w, mode)
        ms.append(m["pr_auc"])
    out["E"]["E3_pooled_reference"] = {"mean_pr_auc": float(np.mean(ms)), "std": float(np.std(ms)),
                                       "note": "Reference only; requires sharing raw data."}
    # E4 cold start: bank with fewest train frauds
    stats = D.get_stats()
    target = min(stats["banks"], key=lambda b: stats["banks"][b]["fraud_train"])
    b = int(target)
    e4 = {}
    for s in seeds:
        df = D.bank_partition("train", b).head(5000)
        c = BankClient(b, df, s)
        rng = np.random.RandomState(s)
        nfeat = len(F.feature_names(mode)); w = np.zeros(nfeat + 1)
        cc = dict(feature_mode=mode, local_epochs=rounds * epochs, clip_norm=None,
                  noise_multiplier=0.0, sampling_frac=1.0, seed=s)
        w_alone = w + fedavg([c.train_round(w, nfeat, cc, 1, rng)])
        others = [x for x in range(5) if x != b]
        w_fed = _quick_train(others + [b], dict(feature_mode=mode, rounds=rounds,
                                                local_epochs=epochs, clip_norm=None,
                                                noise_multiplier=0.0, sampling_frac=1.0), s, stop)
        test = D.load_split("test"); dt = test[test["BankID"] == b]
        Xt = F.transform(dt, mode).values; yt = dt["isFraud"].values
        sa = _sigmoid(Xt @ w_alone[:-1] + w_alone[-1]); sf = _sigmoid(Xt @ w_fed[:-1] + w_fed[-1])
        e4.setdefault("alone", []).append(M.pr_auc(yt, sa))
        e4.setdefault("joined", []).append(M.pr_auc(yt, sf))
    out["E"]["E4_coldstart"] = {"target_bank": b,
                                "alone_mean": float(np.mean(e4["alone"])),
                                "joined_mean": float(np.mean(e4["joined"]))}
    # E5 non-IID: global 5-bank vs each local on each bank test
    w5 = _quick_train([0, 1, 2, 3, 4], dict(feature_mode=mode, rounds=rounds,
                                            local_epochs=epochs, clip_norm=None,
                                            noise_multiplier=0.0, sampling_frac=1.0), seeds[0], stop)
    test = D.load_split("test")
    e5 = {}
    for bt in range(5):
        dt = test[test["BankID"] == bt]
        Xt = F.transform(dt, mode).values; yt = dt["isFraud"].values
        sg = _sigmoid(Xt @ w5[:-1] + w5[-1])
        e5[str(bt)] = {"global_pr_auc": M.pr_auc(yt, sg),
                       "local_pr_auc": e1[str(bt)]["mean_pr_auc"]}
    out["E"]["E5_noniid"] = e5
    # E6 noise sweep (clip from median norm of clean run ~ use 1.0 default)
    e6 = {}
    for nm in [0, 0.05, 0.1, 0.5, 1.0]:
        ms = []
        for s in seeds:
            w = _quick_train([0, 1, 2, 3, 4], dict(feature_mode=mode, rounds=rounds,
                                                   local_epochs=epochs, clip_norm=1.0,
                                                   noise_multiplier=nm), s, stop)
            m, _, _ = _eval(w, mode)
            ms.append(m["pr_auc"])
        e6[str(nm)] = float(np.mean(ms))
    out["E"]["E6_noise"] = e6
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "experiments.json").write_text(json.dumps(out, indent=1))
    db.set_job(jid, result_path=str(RESULTS_DIR / "experiments.json"))
