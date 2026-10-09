"""Priority 1 correctness regression tests (isolated temp DB, synthetic arrays)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tdb(tmp_path, monkeypatch):
    import backend.database as db
    import backend.config as C
    p = tmp_path / "p1.db"
    monkeypatch.setattr(C, "DB_PATH", p)
    monkeypatch.setattr(db, "DB_PATH", p)
    db.init_db()
    return db


def test_add_message_records_payload_evidence(tdb):
    from backend.federated import UpdateMessage
    from backend import database as db
    m = UpdateMessage(1, 1, np.array([0.5, -0.2]), 100, 0.54, 0.54, 0.0)
    db.add_message(9, 1, m)
    rows = db.get_messages(9)
    assert len(rows) == 1
    assert rows[0]["rows_transmitted"] == 0
    assert rows[0]["payload_type"].startswith("numeric_update_array")
    assert "shape" in rows[0]["payload_type"]


def test_add_message_rejects_raw_dataframe(tdb):
    from backend import database as db
    with pytest.raises(ValueError):
        db.add_message(9, 1, pd.DataFrame({"a": [1]}))
    assert db.get_messages(9) == []


def test_threshold_maximizes_f1_on_given_labels():
    from backend.metrics import best_threshold
    y = np.array([0] * 80 + [1] * 20)
    s = np.array([0.1] * 80 + [0.9] * 20)
    thr, f1 = best_threshold(y, s)
    # Contract: threshold is chosen from the labels handed to it (validation),
    # never from held-out test labels; perfect separation gives F1 1.0.
    assert f1 == pytest.approx(1.0)
    assert 0.1 < thr <= 0.9


def test_recall_at_fpr_uses_descending_rank():
    from backend.metrics import recall_at_fpr
    y = np.array([1] * 10 + [0] * 1000)
    s = np.array([0.9] * 10 + [0.1] * 1000)
    assert recall_at_fpr(y, s, 0.01) == 1.0
