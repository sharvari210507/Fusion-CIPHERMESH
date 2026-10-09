"""Privacy audit: verifiable counters of what moved bank -> coordinator.

- raw_rows_sent: incremented ONLY where raw DataFrames are handed to the
  coordinator. The federated path never does this, so it stays 0 — verified
  by test (inspects coordinator inputs) not by assertion alone.
- updates_submitted / accepted / rejected, participating banks, rounds.
- Event timeline (mirrors trust ledger event types).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PrivacyAudit:
    raw_rows_sent: int = 0
    updates_submitted: int = 0
    updates_accepted: int = 0
    updates_rejected: int = 0
    rounds_completed: int = 0
    participants: list = field(default_factory=list)
    events: list = field(default_factory=list)
    last_training_time: str | None = None

    def log(self, kind: str, actor: str, detail: str = "", round_no: int = 0):
        self.events.append({"t": datetime.now(timezone.utc).isoformat(),
                            "event": kind, "actor": actor,
                            "round": round_no, "detail": detail})

    def record_raw_rows(self, n: int, actor: str, round_no: int = 0):
        self.raw_rows_sent += int(n)
        self.log("raw_rows_sent", actor, f"{n} rows", round_no)

    def record_update(self, accepted: bool, actor: str, round_no: int = 0):
        self.updates_submitted += 1
        self.updates_accepted += int(accepted)
        self.updates_rejected += int(not accepted)
        self.log("update_accepted" if accepted else "update_rejected",
                 actor, "", round_no)

    def finish_round(self, banks: list, round_no: int):
        self.rounds_completed = max(self.rounds_completed, round_no)
        self.participants = sorted(set(self.participants) | set(banks))
        self.last_training_time = datetime.now(timezone.utc).isoformat()
        self.log("round_complete", "coordinator", f"banks={banks}", round_no)

    def summary(self) -> dict:
        return {"raw_rows_sent": self.raw_rows_sent,
                "updates_submitted": self.updates_submitted,
                "updates_accepted": self.updates_accepted,
                "updates_rejected": self.updates_rejected,
                "rounds_completed": self.rounds_completed,
                "participants": self.participants,
                "last_training_time": self.last_training_time}

    DISCLAIMER = ("Zero raw rows sent does NOT mean zero privacy risk: "
                  "model updates can still leak information about training rows.")
