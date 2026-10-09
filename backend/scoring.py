"""Transaction scoring: validation, probabilities, contributions."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from .config import MODELS_DIR, TYPES, AMOUNT_MAX
from . import features as F
from . import database as db


def validate_input(d):
    if d.get("type") not in TYPES:
        raise ValueError("type must be one of " + ",".join(TYPES))
    for k in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
        v = d.get(k)
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError(f"{k} must be a number")
        if v < 0 or v > AMOUNT_MAX:
            raise ValueError(f"{k} out of range 0..{AMOUNT_MAX:g}")
    h = d.get("hour", 12)
    if not isinstance(h, (int, float)) or not (0 <= h <= 23):
        raise ValueError("hour must be 0..23")


def _load(job_id):
    p = MODELS_DIR / f"global_run_{job_id}.joblib"
    if not p.exists():
        raise RuntimeError("No active model. Train a run first.")
    return joblib.load(p)


def _sig(z):
    return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))


def score(d, job_id, bank_id, user):
    validate_input(d)
    if bank_id not in range(5):
        raise ValueError("bank_id must be 0..4")
    m = _load(job_id)
    w = m["weights"]; mode = m["feature_mode"]; thr = m.get("threshold", 0.5)
    row = pd.DataFrame([{"step": int(d.get("hour", 12)), "type": d["type"],
                         "amount": d["amount"], "oldbalanceOrg": d["oldbalanceOrg"],
                         "newbalanceOrig": d["newbalanceOrig"],
                         "oldbalanceDest": d["oldbalanceDest"],
                         "newbalanceDest": d["newbalanceDest"]}])
    x = F.transform(row, mode).values[0]
    p_global = _sig(x @ w[:-1] + w[-1])
    contrib = sorted(zip(m["features"], (w[:-1] * x).tolist()),
                     key=lambda t: -abs(t[1]))[:5]
    decision = "FRAUD" if p_global >= thr else "LEGITIMATE"
    import json, sqlite3
    from .config import DB_PATH
    c = sqlite3.connect(str(DB_PATH))
    c.execute("INSERT INTO predictions(user_id,input_json,job_id,bank_id,p_global,p_local,created_at) VALUES(?,?,?,?,?,?,?)",
              (str((user or {}).get("name")), json.dumps(d), int(job_id), int(bank_id),
               float(p_global), None, db.now()))
    c.commit(); c.close()
    db.audit(str((user or {}).get("name")), "score", f"job {job_id} bank {bank_id} p={p_global:.4f}")
    return {"p_global": float(p_global), "p_local": None, "threshold": float(thr),
            "decision": decision, "contributions": contrib}


def example_row(fraud=True):
    from . import data as D
    t = D.load_split("test", columns=["step", "type", "amount", "oldbalanceOrg",
                                      "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"])
    d = t[t["isFraud"] == (1 if fraud else 0)].iloc[0]
    return {"type": str(d["type"]), "amount": float(d["amount"]),
            "oldbalanceOrg": float(d["oldbalanceOrg"]), "newbalanceOrig": float(d["newbalanceOrig"]),
            "oldbalanceDest": float(d["oldbalanceDest"]), "newbalanceDest": float(d["newbalanceDest"]),
            "hour": int(d["step"]) % 24}
