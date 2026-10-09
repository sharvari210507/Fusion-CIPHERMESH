from src.trust_ledger import TrustLedger


def test_ledger_verify_passes():
    tl = TrustLedger()
    tl.append("local_train_start", "Bank A", 1, {"n": 100})
    tl.append("aggregated", "coordinator", 1, {"accepted": 2})
    ok, _ = tl.verify()
    assert ok is True


def test_ledger_detects_tamper():
    tl = TrustLedger()
    tl.append("local_train_start", "Bank A", 1, {"n": 100})
    tl.append("aggregated", "coordinator", 1, {"accepted": 2})
    tl.events[0]["meta"]["n"] = 99999  # attacker edits stored event
    ok, _ = tl.verify()
    assert ok is False
    demo = TrustLedger()
    demo.append("x", "coordinator", 0, {})
    d = demo.tamper_demo()
    assert d["verify_ok"] is False
