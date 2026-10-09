"""Privacy: coordinator must never receive raw rows; metrics safe on rare labels."""
import inspect
import numpy as np
import pandas as pd

import src.coordinator as C
from src.evaluation import evaluate_params
from src.preprocessing import Preprocessor, feature_names
from src.privacy_audit import PrivacyAudit


def test_coordinator_signature_takes_no_raw_rows():
    src = inspect.getsource(C.Coordinator.run_round) + inspect.getsource(C.fedavg)
    assert "DataFrame" not in src  # aggregation path handles numeric params only


def test_raw_rows_counter_stays_zero_in_pipeline():
    from src.pipeline import run_experiment
    out, _, _ = run_experiment(subset_total=2000, n_rounds=1, local_iters=5,
                               force_synthetic=True, save=False)
    assert out["audit"]["raw_rows_sent"] == 0
    assert out["audit"]["updates_submitted"] >= 5


def test_metrics_handle_no_fraud():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, len(feature_names())))
    y = np.zeros(20, dtype=int)
    m = evaluate_params({"coef": np.zeros((1, X.shape[1])), "intercept": np.zeros(1)}, X, y)
    assert m["f1"] is None and m["warning"] is not None and m["n_fraud"] == 0


def test_preprocessing_compatible_across_banks():
    tr = pd.DataFrame({"step": [1, 2], "type": ["PAYMENT", "TRANSFER"], "amount": [10, 20],
                       "oldbalanceOrg": [100, 100], "newbalanceOrig": [90, 80],
                       "oldbalanceDest": [0, 0], "newbalanceDest": [10, 20],
                       "isFlaggedFraud": [0, 0]})
    pre = Preprocessor().fit(tr)
    for t in ["CASH_IN", "PAYMENT", "UNKNOWN_TYPE"]:
        te = tr.copy()
        te["type"] = t
        X = pre.transform(te)
        assert X.shape[1] == len(feature_names()) and np.all(np.isfinite(X))
