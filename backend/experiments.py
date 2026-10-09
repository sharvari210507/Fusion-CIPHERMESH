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


def _quick_train(banks, cfg, seed, stop=None, row_limit=None, record_norms=False):
    """Federated training over the given banks. row_limit caps each bank's train
    rows (cold-start); record_norms returns per-bank update norms of round 1."""
    rng = np.random.RandomState(seed)
    mode = cfg.get("feature_mode", "raw")
    nfeat = len(F.feature_names(mode))
    w = np.zeros(nfeat + 1)
    clients = []
    for b in banks:
        df = D.bank_partition("train", b)
        if row_limit is not None:
            df = df.head(row_limit)
        clients.append(BankClient(b, df, seed))
    c2 = dict(cfg); c2["seed"] = seed
    norms = []
    for r in range(1, cfg.get("rounds", 5) + 1):
        if stop is not None and stop.is_set():
            break
        ups = [c.train_round(w, nfeat, c2, r, rng) for c in clients]
        if record_norms and r == 1:
            norms = [u.norm_before_clip for u in ups]
        w = w + fedavg(ups)
    if record_norms:
        return w, norms
    return w


def _train_local(bank_id, cfg, seed, row_limit=None):
    df = D.bank_partition("train", bank_id)
    if row_limit is not None:
        df = df.head(row_limit)
    c = BankClient(bank_id, df, seed)
    rng = np.random.RandomState(seed)
    nfeat = len(F.feature_names(cfg.get("feature_mode", "raw")))
    w = np.zeros(nfeat + 1)
    w = w + fedavg([c.train_round(w, nfeat, cfg, 1, rng)])
    return w


def _eval(w, mode, thr=0.5, bank=None):
    test = D.load_split("test")
    if bank is not None:
        test = test[test["BankID"] == bank]
    X = F.transform(test, mode).values; y = test["isFraud"].values
    s = _sigmoid(X @ w[:-1] + w[-1])
    return M.full_metrics(y, s, thr), s, y


def run_suite(jid, cfg, stop):
    seeds = cfg.get("seeds", EXP_DEFAULT["seeds"])
    rounds = cfg.get("rounds", 5); epochs = cfg.get("local_epochs", 1)
    out = {"config": cfg, "E": {}}
    mode = "raw"
    # E1 local only: same total epochs as the federated run. Evaluated on the
    # global test split AND on the bank's own test partition. Local models saved.
    e1 = {}
    for b in range(5):
        g, o = [], []
        for s in seeds:
            cc = dict(feature_mode=mode, local_epochs=rounds * epochs, clip_norm=None,
                      noise_multiplier=0.0, sampling_frac=1.0, seed=s)
            w = _train_local(b, cc, s)
            mg, _, _ = _eval(w, mode)
            mo, _, _ = _eval(w, mode, bank=b)
            g.append(mg["pr_auc"]); o.append(mo["pr_auc"])
        e1[str(b)] = {"global_pr_auc_mean": float(np.mean(g)), "global_std": float(np.std(g)),
                      "own_pr_auc_mean": float(np.mean(o)), "own_std": float(np.std(o))}
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"weights": _train_local(b, dict(feature_mode=mode,
                      local_epochs=rounds * epochs, clip_norm=None, noise_multiplier=0.0,
                      sampling_frac=1.0, seed=seeds[0]), seeds[0]),
                     "feature_mode": mode, "bank": b},
                    MODELS_DIR / f"local_bank_{b}.joblib")
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
    # E4 cold start: bank with fewest train frauds, limited to 5,000 rows IN BOTH
    # arms (alone and joined). Records how many fraud rows the limit leaves.
    stats = D.get_stats()
    target = min(stats["banks"], key=lambda b: stats["banks"][b]["fraud_train"])
    b = int(target)
    limited = D.bank_partition("train", b).head(5000)
    e4fraud = int(limited["isFraud"].sum())
    e4 = {}
    for s in seeds:
        cc = dict(feature_mode=mode, local_epochs=rounds * epochs, clip_norm=None,
                  noise_multiplier=0.0, sampling_frac=1.0, seed=s)
        w_alone = _train_local(b, cc, s, row_limit=5000)
        full = [x for x in range(5) if x != b]
        # joined: target contributes only its limited rows alongside 4 full banks
        fed_cfg = dict(feature_mode=mode, rounds=rounds, local_epochs=epochs,
                       clip_norm=None, noise_multiplier=0.0, sampling_frac=1.0)
        rng = np.random.RandomState(s)
        nfeat = len(F.feature_names(mode)); w = np.zeros(nfeat + 1)
        clients = []
        for bb in full + [b]:
            df = D.bank_partition("train", bb)
            if bb == b:
                df = df.head(5000)
            clients.append(BankClient(bb, df, s))
        c2 = dict(fed_cfg); c2["seed"] = s
        for r in range(1, rounds + 1):
            if stop is not None and stop.is_set():
                break
            w = w + fedavg([c.train_round(w, nfeat, c2, r, rng) for c in clients])
        w_fed = w
        ma, _, _ = _eval(w_alone, mode, bank=b)
        mf, _, _ = _eval(w_fed, mode, bank=b)
        mg, _, _ = _eval(w_fed, mode)
        e4.setdefault("alone_own", []).append(ma["pr_auc"])
        e4.setdefault("joined_own", []).append(mf["pr_auc"])
        e4.setdefault("joined_global", []).append(mg["pr_auc"])
    out["E"]["E4_coldstart"] = {"target_bank": b, "limited_rows": 5000,
                                "limited_fraud_rows": e4fraud,
                                "alone_own_mean": float(np.mean(e4["alone_own"])),
                                "joined_own_mean": float(np.mean(e4["joined_own"])),
                                "joined_global_mean": float(np.mean(e4["joined_global"]))}
    # E5 Bank Divergence Radar: the final 5-bank global model versus each bank's
    # OWN local-only model, both evaluated on THAT bank's test partition.
    e5 = {}
    for bt in range(5):
        loc, glo = [], []
        for s in seeds:
            cc = dict(feature_mode=mode, local_epochs=rounds * epochs, clip_norm=None,
                      noise_multiplier=0.0, sampling_frac=1.0, seed=s)
            wl = _train_local(bt, cc, s)
            ml, _, _ = _eval(wl, mode, bank=bt)
            loc.append(ml["pr_auc"])
        w5 = _quick_train([0, 1, 2, 3, 4], dict(feature_mode=mode, rounds=rounds,
                                                local_epochs=epochs, clip_norm=None,
                                                noise_multiplier=0.0, sampling_frac=1.0), seeds[0], stop)
        mg, _, _ = _eval(w5, mode, bank=bt)
        e5[str(bt)] = {"global_pr_auc": float(mg["pr_auc"]),
                       "local_own_mean": float(np.mean(loc)),
                       "local_own_std": float(np.std(loc)),
                       "delta": float(mg["pr_auc"] - np.mean(loc))}
    out["E"]["E5_noniid"] = e5
    # E6 noise sweep: clip norm measured (median round-1 update norm of a clean
    # probe run), not guessed; recorded in output.
    _, probe_norms = _quick_train([0, 1, 2, 3, 4], dict(feature_mode=mode, rounds=1,
                                  local_epochs=epochs, clip_norm=None,
                                  noise_multiplier=0.0, sampling_frac=1.0),
                                  seeds[0], stop, record_norms=True)
    clip_used = float(np.median(probe_norms)) if probe_norms else 1.0
    e6 = {}
    for nm in [0, 0.05, 0.1, 0.5, 1.0]:
        ms = []
        for s in seeds:
            w = _quick_train([0, 1, 2, 3, 4], dict(feature_mode=mode, rounds=rounds,
                                                   local_epochs=epochs, clip_norm=clip_used,
                                                   noise_multiplier=nm), s, stop)
            m, _, _ = _eval(w, mode)
            ms.append(m["pr_auc"])
        e6[str(nm)] = float(np.mean(ms))
    out["E"]["E6_noise"] = {"clip_norm_measured": clip_used, "sweep": e6}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "experiments.json").write_text(json.dumps(out, indent=1))
    db.set_job(jid, result_path=str(RESULTS_DIR / "experiments.json"))
