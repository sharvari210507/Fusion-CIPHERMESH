"""Bank client, FedAvg, run loop. Trust boundary: clients hold DataFrames;
aggregator sees only UpdateMessage (numeric array + metadata)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from . import features as F
from .privacy import clip_update, add_noise


@dataclass
class UpdateMessage:
    bank_id: int
    round: int
    update: np.ndarray
    n_samples: int
    norm_before_clip: float
    norm_after_clip: float
    noise_std: float


def validate_message(msg):
    if isinstance(msg, pd.DataFrame):
        raise ValueError("Raw DataFrames must never enter the aggregator.")
    if not isinstance(msg, UpdateMessage):
        raise ValueError("Unexpected payload type.")
    u = msg.update
    if not isinstance(u, np.ndarray) or u.dtype.kind not in "fiu":
        raise ValueError("Update must be a numeric array.")
    if u.ndim != 1 or not np.all(np.isfinite(u)):
        raise ValueError("Invalid update values.")
    if not isinstance(msg.n_samples, int) or msg.n_samples <= 0:
        raise ValueError("Invalid n_samples.")
    return True


def fedavg(updates):
    for m in updates:
        validate_message(m)
    total = sum(m.n_samples for m in updates)
    agg = sum(m.update * m.n_samples for m in updates) / total
    return agg


def _split_train_val(df, seed):
    fraud = df[df["isFraud"] == 1]; non = df[df["isFraud"] == 0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    nval = max(1, int(0.1 * len(df)))
    val = df.iloc[idx[:nval]]; tr = df.iloc[idx[nval:]]
    return tr, val


class BankClient:
    def __init__(self, bank_id, train_df, seed=42):
        self.bank_id = bank_id
        self.seed = seed
        self.train_df, self.val_df = _split_train_val(
            train_df.reset_index(drop=True), seed + bank_id)

    def train_round(self, global_w, n_features, cfg, round_no, rng):
        mode = cfg.get("feature_mode", "raw")
        X = F.transform(self.train_df, mode).values
        y = self.train_df["isFraud"].values
        frac = cfg.get("sampling_frac", 1.0)
        if frac < 1.0:
            r = np.random.RandomState(self.seed + round_no)
            keep = np.ones(len(y), bool)
            nf = np.where(y == 0)[0]
            drop = r.choice(nf, size=int(len(nf) * (1 - frac)), replace=False)
            keep[drop] = False
            X, y = X[keep], y[keep]
        coef0 = global_w[:-1].reshape(1, -1); int0 = global_w[-1:].reshape(1, -1) \
            if global_w[-1:].shape else global_w[-1:]
        clf = SGDClassifier(loss="log_loss", max_iter=cfg.get("local_epochs", 1),
                            learning_rate="optimal", class_weight="balanced",
                            random_state=self.seed + round_no, warm_start=False)
        clf.fit(X, y, coef_init=coef0, intercept_init=int0.reshape(-1))
        local_w = np.concatenate([clf.coef_.ravel(), clf.intercept_.ravel()]).astype(float)
        delta = local_w - global_w
        delta, nb, na = clip_update(delta, cfg.get("clip_norm"))
        delta, std = add_noise(delta, cfg.get("clip_norm"),
                               cfg.get("noise_multiplier", 0.0), rng)
        return UpdateMessage(self.bank_id, round_no, delta.astype(np.float64),
                             int(len(y)), nb, na, std)

    def val_data(self, mode="raw"):
        return F.transform(self.val_df, mode).values, self.val_df["isFraud"].values
