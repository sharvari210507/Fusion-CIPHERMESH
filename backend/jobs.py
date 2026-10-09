"""Background job manager + training run loop + startup routine."""
import json
import threading
import traceback
from pathlib import Path
import joblib
import numpy as np

from . import database as db
from . import data as D
from . import features as F
from . import metrics as M
from .federated import BankClient, fedavg
from .config import MODELS_DIR, RESULTS_DIR, DEFAULT_RUN


_manager = None


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _validate_cfg(cfg):
    banks = cfg.get("banks", [0, 1, 2, 3, 4])
    if not banks or any(b not in range(5) for b in banks):
        raise ValueError("banks must be a non-empty subset of 0..4")
    if not (1 <= cfg.get("rounds", 5) <= 50):
        raise ValueError("rounds must be 1..50")
    if not (1 <= cfg.get("local_epochs", 1) <= 20):
        raise ValueError("local_epochs must be 1..20")
    if cfg.get("feature_mode") not in ("raw", "engineered"):
        raise ValueError("feature_mode must be raw/engineered")
    cn = cfg.get("clip_norm")
    if cn is not None and not (0.01 <= cn <= 100):
        raise ValueError("clip_norm out of range")
    nm = cfg.get("noise_multiplier", 0.0)
    if not (0 <= nm <= 5):
        raise ValueError("noise_multiplier out of range")
    if nm > 0 and cn is None:
        raise ValueError("noise requires clip_norm")
    if not (0.01 <= cfg.get("sampling_frac", 1.0) <= 1.0):
        raise ValueError("sampling_frac must be 0.01..1.0")


class Manager:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue = []
        self.current = None
        self.stop = threading.Event()

    def submit(self, kind, cfg, user):
        if kind == "training":
            _validate_cfg(cfg)
            from .auth import require_admin
            require_admin(user)
        jid = db.create_job(kind, cfg, (user or {}).get("name", "system"))
        db.audit((user or {}).get("name", "system"), f"{kind}_start", f"job {jid}")
        with self.lock:
            self.queue.append(jid)
            if self.current is None:
                threading.Thread(target=self._loop, daemon=True).start()
        return jid

    def cancel(self, jid, user):
        from .auth import require_admin
        require_admin(user)
        with self.lock:
            if jid in self.queue:
                self.queue.remove(jid)
                db.set_job(jid, status="cancelled", finished_at=db.now())
                return True
            if self.current == jid:
                self.stop.set()
                return True
        return False

    def status(self):
        with self.lock:
            return {"current": self.current, "queued": list(self.queue)}

    def _loop(self):
        while True:
            with self.lock:
                if not self.queue:
                    self.current = None
                    return
                jid = self.queue.pop(0)
                self.current = jid
                self.stop.clear()
            try:
                job = db.get_job(jid)
                db.set_job(jid, status="running", started_at=db.now())
                if job["kind"] == "training":
                    _run_training(jid, json.loads(job["config_json"]), self.stop)
                else:
                    from . import experiments as E
                    E.run_suite(jid, json.loads(job["config_json"]), self.stop)
                if self.stop.is_set():
                    db.set_job(jid, status="cancelled", finished_at=db.now())
                else:
                    db.set_job(jid, status="completed", finished_at=db.now())
            except Exception as e:
                db.set_job(jid, status="failed", finished_at=db.now(), error=str(e)[:500])
            finally:
                with self.lock:
                    if self.current == jid:
                        self.current = None


def _run_training(jid, cfg, stop):
    from .federated import validate_message
    rng = np.random.RandomState(cfg.get("seed", 42))
    mode = cfg["feature_mode"]
    nfeat = len(F.feature_names(mode))
    w = np.zeros(nfeat + 1)
    clients = [BankClient(b, D.bank_partition("train", b), cfg.get("seed", 42))
               for b in cfg["banks"]]
    test = D.load_split("test")
    Xte_full = F.transform(test, mode).values; yte_full = test["isFraud"].values
    # pooled validation for threshold
    import pandas as pd
    val_frames = [c.val_df for c in clients]
    Xv = F.transform(pd.concat(val_frames), mode).values if val_frames else None
    thr = 0.5
    history = []
    for r in range(1, cfg["rounds"] + 1):
        if stop.is_set():
            break
        updates = [c.train_round(w, nfeat, cfg, r, rng) for c in clients]
        for m in updates:
            validate_message(m)
            db.add_message(jid, r, m)
        w = w + fedavg(updates)
        s = _sigmoid(Xte_full @ w[:-1] + w[-1])
        if Xv is not None:
            sv = _sigmoid(Xv @ w[:-1] + w[-1])
            yv = pd.concat(val_frames)["isFraud"].values
            thr, _ = M.best_threshold(yv, sv)
        m = M.full_metrics(yte_full, s, thr)
        db.add_round_metric(jid, r, m["pr_auc"], m["recall_at_1pct_fpr"], m["f1"])
        history.append({"round": r, **m, "threshold": thr})
    MODELS_DIR.mkdir(parents=True, exist_ok=True); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"weights": w, "feature_mode": mode, "threshold": thr,
                 "features": F.feature_names(mode)},
                MODELS_DIR / f"global_run_{jid}.joblib")
    (RESULTS_DIR / f"run_{jid}.json").write_text(json.dumps(
        {"job_id": jid, "config": cfg, "threshold": thr, "history": history}, indent=1))
    # per-bank test metrics
    per_bank = {}
    for b in range(5):
        d = test[test["BankID"] == b]
        s = _sigmoid(F.transform(d, mode).values @ w[:-1] + w[-1])
        per_bank[str(b)] = M.full_metrics(d["isFraud"].values, s, thr)
    (RESULTS_DIR / f"run_{jid}_perbank.json").write_text(json.dumps(per_bank, indent=1))
    if db.get_setting("active_model") is None:
        db.set_setting("active_model", jid)


def get_manager():
    global _manager
    if _manager is None:
        _manager = Manager()
    return _manager


def startup():
    from .config import AUTO_START_JOBS
    db.init_db()
    for j in db.list_jobs(limit=100):
        if j["status"] == "running":
            db.set_job(j["id"], status="interrupted", finished_at=db.now())
    if not D.is_cached():
        try:
            D.download_and_cache()
        except Exception:
            pass
    else:
        try:
            D.compute_stats()
        except Exception:
            pass
    if AUTO_START_JOBS and D.is_cached():
        jobs = db.list_jobs(limit=100)
        if not any(j["status"] == "completed" and j["kind"] == "training" for j in jobs):
            mgr = get_manager()
            if mgr.current is None and not mgr.queue:
                mgr.submit("training", dict(DEFAULT_RUN), {"name": "system", "role": "admin"})
