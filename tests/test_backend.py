import numpy as np
import pandas as pd


def test_fedavg_weighted():
    from backend.federated import fedavg, UpdateMessage
    a = UpdateMessage(0, 1, np.array([1.0, 2.0]), 100, 1, 1, 0.0)
    b = UpdateMessage(1, 1, np.array([3.0, 4.0]), 300, 1, 1, 0.0)
    r = fedavg([a, b])
    assert np.allclose(r, np.array([2.5, 3.5]))


def test_clip_norm():
    from backend.privacy import clip_update
    d, nb, na = clip_update(np.array([3.0, 4.0]), 1.0)
    assert na <= 1.0 + 1e-9


def test_noise_std():
    from backend.privacy import add_noise
    rng = np.random.RandomState(0)
    xs = np.stack([add_noise(np.zeros(200), 1.0, 0.5, rng)[0] for _ in range(30)])
    assert abs(xs.std() - 0.5) < 0.1


def test_reject_dataframe():
    from backend.federated import validate_message
    try:
        validate_message(pd.DataFrame({"a": [1]}))
    except ValueError:
        return
    raise AssertionError("DataFrame not rejected")


def test_reject_nan():
    from backend.federated import validate_message, UpdateMessage
    try:
        validate_message(UpdateMessage(0, 1, np.array([np.nan]), 10, 1, 1, 0.0))
    except ValueError:
        return
    raise AssertionError("NaN not rejected")


def test_feature_order():
    from backend.features import transform
    df = pd.DataFrame({"step": [1, 2], "type": ["CASH_IN", "TRANSFER"], "amount": [10, 20],
                       "oldbalanceOrg": [10, 20], "newbalanceOrig": [0, 0],
                       "oldbalanceDest": [0, 0], "newbalanceDest": [10, 20]})
    a = list(transform(df[df["type"] == "CASH_IN"], "raw").columns)
    b = list(transform(df, "raw").columns)
    assert a == b


def test_scoring_validation():
    from backend.scoring import validate_input
    try:
        validate_input({"type": "HACK", "amount": -1, "oldbalanceOrg": 0,
                        "newbalanceOrig": 0, "oldbalanceDest": 0, "newbalanceDest": 0})
    except ValueError:
        return
    raise AssertionError("bad input not rejected")


def test_recall_at_fpr_known():
    from backend.metrics import recall_at_fpr
    # perfect ranking: all fraud scored above all legit -> recall 1.0 at 1% FPR
    y = np.array([1] * 10 + [0] * 1000)
    s = np.array([0.9] * 10 + [0.1] * 1000)
    assert recall_at_fpr(y, s, 0.01) == 1.0
    # worst ranking: fraud at bottom -> recall 0.0
    s2 = np.array([0.1] * 10 + [0.9] * 1000)
    assert recall_at_fpr(y, s2, 0.01) == 0.0


def test_lockout_and_roles(tmp_path, monkeypatch):
    import backend.database as db
    import backend.config as C
    monkeypatch.setattr(C, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    from backend import auth as A
    ok, _ = A.register("tester1", "longpassword1")
    assert ok
    for _ in range(5):
        A.login_local("tester1", "wrongpassword")
    ok, _ = A.login_local("tester1", "longpassword1")
    assert not ok  # locked
    try:
        A.require_admin({"name": "x", "role": "analyst"})
    except PermissionError:
        return
    raise AssertionError("analyst passed admin check")
