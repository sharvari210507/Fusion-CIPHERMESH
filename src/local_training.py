"""Local training: every bank trains the SAME architecture from the SAME start.

- Model: SGDClassifier(loss="log_loss") — logistic regression via SGD.
- Each round the bank receives the current global (coef, intercept), installs
  them, then runs `local_iters` SGD epochs (partial_fit) on its own rows.
- Only parameters + counts leave the bank. Raw rows never leave.
- Single-class fallback: if a bank's train split has only one class,
  `partial_fit` is still called with classes_=[0,1] so shapes stay compatible;
  a warning is recorded instead of crashing.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from src.preprocessing import validate_feature_matrix


def new_model(seed: int = C.RANDOM_SEED, max_iter: int = C.LOCAL_MAX_ITER) -> SGDClassifier:
    return SGDClassifier(
        loss=C.MODEL_PARAMS["loss"], penalty=C.MODEL_PARAMS["penalty"],
        alpha=C.MODEL_PARAMS["alpha"], learning_rate=C.MODEL_PARAMS["learning_rate"],
        eta0=C.MODEL_PARAMS["eta0"], max_iter=1, tol=None,
        warm_start=False, random_state=seed)


def _ensure_started(model: SGDClassifier, n_features: int) -> SGDClassifier:
    if not hasattr(model, "coef_"):
        model.classes_ = np.array([0, 1])
        model.coef_ = np.zeros((1, n_features))
        model.intercept_ = np.zeros(1)
        model.t_ = 1
    return model


def set_params(model: SGDClassifier, params: dict) -> SGDClassifier:
    n_features = np.asarray(params["coef"]).shape[1]
    _ensure_started(model, n_features)
    model.coef_ = np.asarray(params["coef"], dtype=float).copy()
    model.intercept_ = np.asarray(params["intercept"], dtype=float).copy()
    return model


def get_params(model: SGDClassifier) -> dict:
    return {"coef": np.asarray(model.coef_, dtype=float).copy(),
            "intercept": np.asarray(model.intercept_, dtype=float).copy()}


def train_local(X: np.ndarray, y: np.ndarray, init_params: dict | None = None,
                local_iters: int = C.LOCAL_MAX_ITER, seed: int = C.RANDOM_SEED):
    """Train one bank locally starting from init_params. Returns update dict."""
    validate_feature_matrix(X, context="train_local")
    y = np.asarray(y).astype(int)
    model = new_model(seed)
    n_features = X.shape[1]
    _ensure_started(model, n_features)
    if init_params is not None:
        set_params(model, init_params)
    single_class = len(np.unique(y)) < 2
    warning = None
    if single_class:
        warning = "single-class train split; partial_fit with classes_=[0,1], weak update"
    for _ in range(max(1, int(local_iters))):
        model.partial_fit(X, y, classes=np.array([0, 1]))
    p = get_params(model)
    return {"coef": p["coef"], "intercept": p["intercept"], "n": int(len(X)),
            "meta": {"seed": seed, "local_iters": int(local_iters),
                     "single_class": bool(single_class)},
            "warning": warning}


# ---- backward compatibility with v1 modules (dataset.py / dashboard.py) ----
def train_local_model(X_train, y_train, X_val=None, y_val=None,
                      clip_norm=None, noise_multiplier=0.0, random_state=42):
    """Legacy wrapper: trains from zero init, applies clipping+noise to weights."""
    from src.update_protection import clip_update, add_noise
    upd = train_local(np.asarray(X_train, float), np.asarray(y_train, int),
                      init_params=None, local_iters=50, seed=random_state)
    g0 = {"coef": np.zeros_like(upd["coef"]), "intercept": np.zeros_like(upd["intercept"])}
    upd, _ = clip_update(upd, g0, clip_norm)
    upd = add_noise(upd, clip_norm, noise_multiplier, np.random.default_rng(random_state))
    from src.evaluation import evaluate_params
    tr = evaluate_params({"coef": upd["coef"], "intercept": upd["intercept"]},
                         np.asarray(X_train, float), np.asarray(y_train, int))
    val = evaluate_params({"coef": upd["coef"], "intercept": upd["intercept"]},
                          np.asarray(X_val, float), np.asarray(y_val, int)) \
        if X_val is not None else None
    import copy as _copy
    model = new_model(random_state)
    set_params(model, {"coef": upd["coef"], "intercept": upd["intercept"]})
    return {"model": model, "weights": {"coef": upd["coef"], "intercept": upd["intercept"]},
            "train_metrics": {k: (tr[k] if tr[k] is not None else 0) for k in
                              ("precision", "recall", "f1", "pr_auc")},
            "val_metrics": ({k: (val[k] if val[k] is not None else 0) for k in
                             ("precision", "recall", "f1", "pr_auc")} if val else None),
            "n_samples": int(len(X_train))}


def aggregate_weights(weight_list, sample_counts):
    from src.fedavg import fedavg as _fed
    g, _, _, _ = _fed(
        [{"coef": w["coef"], "intercept": w["intercept"], "n": n, "meta": {}}
         for w, n in zip(weight_list, sample_counts)],
        tuple(np.asarray(weight_list[0]["coef"]).shape),
        tuple(np.asarray(weight_list[0]["intercept"]).shape))
    return g


def set_model_weights(model, weights):
    return set_params(model, weights)


def evaluate_model(model, X_test, y_test):
    from src.evaluation import evaluate_params as _ev
    m = _ev({"coef": model.coef_, "intercept": model.intercept_},
            np.asarray(X_test, float), np.asarray(y_test, int))
    return {k: (m[k] if m[k] is not None else 0) for k in
            ("precision", "recall", "f1", "pr_auc")} | {"n_samples": m["n"]}
