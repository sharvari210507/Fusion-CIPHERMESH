"""Priority 2/5: alert service tests (isolated temp DB, synthetic tokens)."""
import pytest

U = {"name": "tester", "role": "analyst"}
T1 = "a" * 32
T2 = "b" * 32


@pytest.fixture
def adb(tmp_path, monkeypatch):
    import backend.database as db
    import backend.config as C
    p = tmp_path / "alerts.db"
    monkeypatch.setattr(C, "DB_PATH", p)
    monkeypatch.setattr(db, "DB_PATH", p)
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    db.init_db()
    return db


def test_publish_and_match(adb):
    from backend import alerts as A
    sid, created = A.publish(0, "mule_recipient", 0.9, T1, "global_run_3", "demo", U)
    assert created and sid.startswith("sig_")
    hits = A.match(T1, 1, U)
    assert len(hits) == 1
    assert hits[0]["source_bank"] == 0
    assert "not a guilt" in hits[0]["reason"]
    assert hits[0]["latency_ms"] >= 0


def test_different_actor_no_match(adb):
    from backend import alerts as A
    A.publish(0, "mule_recipient", 0.9, T1, "v", "d", U)
    assert A.match(T2, 1, U) == []


def test_same_bank_not_visible(adb):
    from backend import alerts as A
    A.publish(0, "mule_recipient", 0.9, T1, "v", "d", U)
    assert A.match(T1, 0, U) == []


def test_expired_signal(adb):
    from backend import alerts as A
    sid, _ = A.publish(0, "mule_recipient", 0.9, T1, "v", "d", U, ttl_hours=-1)
    assert A.expire_due() >= 1
    assert A.match(T1, 1, U) == []


def test_duplicate_signal(adb):
    from backend import alerts as A
    sid1, c1 = A.publish(0, "mule_recipient", 0.9, T1, "v", "d", U)
    sid2, c2 = A.publish(0, "mule_recipient", 0.8, T1, "v", "d", U)
    assert c1 is True and c2 is False and sid1 == sid2


def test_invalid_payloads(adb):
    import pandas as pd
    from backend import alerts as A
    with pytest.raises(ValueError):
        A.publish(0, "mule_recipient", 0.9, pd.DataFrame({"a": [1]}), "v", "d", U)
    with pytest.raises(ValueError):
        A.publish(9, "mule_recipient", 0.9, T1, "v", "d", U)
    with pytest.raises(ValueError):
        A.publish(0, "nope", 0.9, T1, "v", "d", U)
    with pytest.raises(ValueError):
        A.publish(0, "mule_recipient", 9.9, T1, "v", "d", U)
    with pytest.raises(ValueError):
        A.match("", 1, U)


def test_unauthorized(adb):
    from backend import alerts as A
    with pytest.raises(PermissionError):
        A.publish(0, "mule_recipient", 0.9, T1, "v", "d", None)
    with pytest.raises(PermissionError):
        A.match(T1, 1, None)


def test_token_deterministic_and_salted(adb):
    from backend import alerts as A
    assert A.entity_token("C123") == A.entity_token("C123")
    assert A.entity_token("C123") != A.entity_token("C124")
    with pytest.raises(ValueError):
        A.entity_token("")
