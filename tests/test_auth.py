"""Phase 2: authentication and session security tests (isolated temp DB)."""
import pytest


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import backend.database as db
    import backend.config as C
    p = tmp_path / "auth_test.db"
    monkeypatch.setattr(C, "DB_PATH", p)
    monkeypatch.setattr(db, "DB_PATH", p)
    import backend.auth as A
    monkeypatch.setattr(A, "_c", lambda: __import__("sqlite3").connect(str(p), timeout=30))
    db.init_db()
    return db


def test_valid_login(isolated_db):
    from backend import auth as A
    ok, _ = A.register("analyst1", "longpassword1")
    assert ok
    ok, msg = A.login_local("analyst1", "longpassword1")
    assert ok, msg


def test_invalid_login_rejected(isolated_db):
    from backend import auth as A
    A.register("analyst2", "longpassword2")
    assert A.login_local("analyst2", "wrongpassword!")[0] is False
    assert A.login_local("nosuchuser", "whateverpass1")[0] is False


def test_password_not_plaintext(isolated_db):
    import sqlite3
    import backend.config as C
    from backend import auth as A
    A.register("analyst3", "longpassword3")
    r = sqlite3.connect(str(C.DB_PATH)).execute(
        "SELECT password_hash FROM users WHERE username='analyst3'").fetchone()
    assert r[0] != "longpassword3"
    assert r[0].startswith("$2")


def test_default_role_analyst(isolated_db):
    import sqlite3
    import backend.config as C
    from backend import auth as A
    A.register("analyst4", "longpassword4")
    r = sqlite3.connect(str(C.DB_PATH)).execute(
        "SELECT role FROM users WHERE username='analyst4'").fetchone()
    assert r[0] == "analyst"


def test_analyst_cannot_start_training(isolated_db):
    from backend.jobs import Manager
    from backend import auth as A
    A.register("analyst5", "longpassword5")
    m = Manager()
    cfg = {"banks": [0], "rounds": 1, "local_epochs": 1, "feature_mode": "raw",
           "clip_norm": None, "noise_multiplier": 0.0, "sampling_frac": 1.0, "seed": 1}
    with pytest.raises(PermissionError):
        m.submit("training", cfg, {"name": "analyst5", "role": "analyst"})


def test_logout_clears_session(isolated_db):
    import streamlit as st
    from backend import auth as A
    A.register("analyst6", "longpassword6")
    A.login_local("analyst6", "longpassword6")
    assert A.get_page_user() is not None
    A.logout()
    assert A.get_page_user() is None


def test_demotion_takes_effect_without_relogin(isolated_db):
    import sqlite3
    import backend.config as C
    from backend import auth as A
    A.register("analyst7", "longpassword7")
    A.login_local("analyst7", "longpassword7")
    c = sqlite3.connect(str(C.DB_PATH))
    c.execute("UPDATE users SET role='admin' WHERE username='analyst7'")
    c.commit(); c.close()
    assert A.get_page_user()["role"] == "admin"
    c = sqlite3.connect(str(C.DB_PATH))
    c.execute("UPDATE users SET role='analyst' WHERE username='analyst7'")
    c.commit(); c.close()
    assert A.get_page_user()["role"] == "analyst"
