"""Evaluation: local-only vs federated on HELD-OUT test rows.

Metrics: precision, recall, F1, PR-AUC (average_precision — right for ~0.1%
fraud), confusion matrix, fraud support count. Accuracy is reported but never
trusted alone. Missing positives -> metric=None + warning (never fabricated).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support)

from src.local_training import set_params


def score_probs(coef: np.ndarray, intercept: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = X @ np.asarray(coef).reshape(-1) + float(np.asarray(intercept).reshape(-1)[0])
    return 1.0 / (1.0 + np.exp(-z))


def evaluate_params(params: dict, X: np.ndarray, y: np.ndarray,
                    threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    proba = score_probs(params["coef"], params["intercept"], np.asarray(X, dtype=float))
    pred = (proba >= threshold).astype(int)
    out = {"n": int(len(y)), "n_fraud": int(y.sum()), "threshold": float(threshold),
           "warning": None}
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    if out["n_fraud"] == 0:
        out.update(precision=None, recall=None, f1=None, pr_auc=None,
                   warning="no fraud rows in evaluation set; metrics undefined")
        return out
    if len(np.unique(y)) < 2:
        out.update(precision=None, recall=None, f1=None, pr_auc=None,
                   warning="single class in evaluation set")
        return out
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try:
        ap = float(average_precision_score(y, proba))
    except Exception:
        ap = None
    out.update(precision=float(p), recall=float(r), f1=float(f), pr_auc=ap)
    return out


def evaluate_sklearn(model: SGDClassifier, X, y, threshold: float = 0.5) -> dict:
    return evaluate_params({"coef": model.coef_, "intercept": model.intercept_},
                           X, y, threshold)
