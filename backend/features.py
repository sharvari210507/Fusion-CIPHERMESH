import numpy as np
import pandas as pd

from .config import TYPES

FEATURES_RAW = ["log_amount", "log_ob_org", "log_nb_org", "log_ob_dst",
                "log_nb_dst", "hour"] + [f"type_{t}" for t in TYPES]
FEATURES_ENG = FEATURES_RAW + ["err_orig", "err_dest"]


def _slog1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def transform(df, mode="raw"):
    df = df.copy()
    out = pd.DataFrame(index=df.index)
    out["log_amount"] = np.log1p(df["amount"].clip(lower=0))
    for c, n in [("oldbalanceOrg", "log_ob_org"), ("newbalanceOrig", "log_nb_org"),
                 ("oldbalanceDest", "log_ob_dst"), ("newbalanceDest", "log_nb_dst")]:
        out[n] = np.log1p(df[c].clip(lower=0))
    out["hour"] = (df["step"] % 24) / 23.0
    for t in TYPES:
        out[f"type_{t}"] = (df["type"] == t).astype(np.float32)
    if mode == "engineered":
        out["err_orig"] = _slog1p(df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"])
        out["err_dest"] = _slog1p(df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"])
    cols = FEATURES_ENG if mode == "engineered" else FEATURES_RAW
    return out[cols].astype(np.float32)


def feature_names(mode="raw"):
    return FEATURES_ENG if mode == "engineered" else FEATURES_RAW
