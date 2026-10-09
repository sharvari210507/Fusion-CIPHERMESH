"""End-to-end smoke test + save/load consistency (small synthetic run)."""
import os
import numpy as np


def test_end_to_end_synthetic():
    from src.pipeline import run_experiment
    out, coord, pre = run_experiment(subset_total=3000, n_rounds=2, local_iters=5,
                                     force_synthetic=True, save=False)
    assert len(out["history"]) == 2
    assert "0" in out["divergence"]  # per-bank local vs federated present
    ok, _ = coord.ledger.verify()
    assert ok is True


def test_save_load_consistency(tmp_path):
    from src.preprocessing import Preprocessor
    import pandas as pd
    tr = pd.DataFrame({"step": [1, 2, 3], "type": ["PAYMENT", "DEBIT", "TRANSFER"],
                       "amount": [10, 20, 30], "oldbalanceOrg": [100, 200, 300],
                       "newbalanceOrig": [90, 180, 270], "oldbalanceDest": [0, 0, 0],
                       "newbalanceDest": [10, 20, 30], "isFlaggedFraud": [0, 0, 0]})
    pre = Preprocessor().fit(tr)
    p = str(tmp_path / "pre.pkl")
    pre.save(p)
    pre2 = Preprocessor.load(p)
    np.testing.assert_allclose(pre.transform(tr), pre2.transform(tr))
    # model params round-trip through npz
    params = {"coef": np.array([[0.1, 0.2]]), "intercept": np.array([0.3])}
    f = str(tmp_path / "m.npz")
    np.savez(f, coef=params["coef"], intercept=params["intercept"])
    d = np.load(f)
    np.testing.assert_allclose(d["coef"], params["coef"])
