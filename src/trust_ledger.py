"""Tamper-evident hash-linked trust ledger (SHA-256).

Each event: seq, timestamp, event type, actor, round, metadata (non-sensitive
only — never raw rows or full parameters, only norms/counts), prev_hash, hash.
verify() recomputes the chain. tamper_demo() flips one stored event to show
verification failing. A passing chain proves the LOG is intact, not that
updates were honest.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone


def _canonical(ev: dict) -> str:
    payload = {"seq": ev["seq"], "ts": ev["ts"], "event": ev["event"],
               "actor": ev["actor"], "round": ev["round"], "meta": ev["meta"],
               "prev": ev["prev"]}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash(ev: dict) -> str:
    return hashlib.sha256(_canonical(ev).encode()).hexdigest()


class TrustLedger:
    def __init__(self):
        self.events: list[dict] = []

    def append(self, event: str, actor: str, round_no: int = 0, meta: dict | None = None) -> dict:
        meta = meta or {}
        ev = {"seq": len(self.events),
              "ts": datetime.now(timezone.utc).isoformat(),
              "event": event, "actor": actor, "round": int(round_no),
              "meta": copy.deepcopy(meta),
              "prev": self.events[-1]["hash"] if self.events else "GENESIS"}
        ev["hash"] = _hash(ev)
        self.events.append(ev)
        return ev

    def verify(self) -> tuple[bool, str]:
        prev = "GENESIS"
        for i, ev in enumerate(self.events):
            if ev["seq"] != i or ev["prev"] != prev:
                return False, f"chain break at seq {i}"
            if _hash(ev) != ev["hash"]:
                return False, f"hash mismatch at seq {i} ({ev['event']})"
            prev = ev["hash"]
        return True, f"ok ({len(self.events)} events)"

    def tamper_demo(self) -> dict:
        """Deliberately modify a copy of event #0 and show verify() failing."""
        if not self.events:
            return {"error": "ledger empty"}
        tampered = copy.deepcopy(self.events)
        tampered[0]["meta"]["tampered"] = True  # hash NOT recomputed
        ledger2 = TrustLedger()
        ledger2.events = tampered
        ok, msg = ledger2.verify()
        return {"tampered_seq": 0, "verify_ok": ok, "verify_msg": msg,
                "expected": "verification must FAIL after modification"}

    def table(self) -> list[dict]:
        return [{"seq": e["seq"], "ts": e["ts"], "event": e["event"],
                 "actor": e["actor"], "round": e["round"],
                 "hash": e["hash"][:12] + "…", "prev": str(e["prev"])[:12] + "…",
                 "meta": json.dumps(e["meta"])[:120]} for e in self.events]
