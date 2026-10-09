"""Real-time cross-bank risk-signal service (isolated from federated training).

Trust model: banks never share raw transactions, account names or raw account
identifiers. A signal carries only a scenario-scoped pseudonymous entity token:
HMAC-SHA256(recipient_id, deployment_salt), truncated. The salt is a random
per-deployment secret held server-side; tokens are meaningless off this
deployment and are rotated by re-provisioning. Plain hashing of predictable
identifiers is NOT claimed as sufficient privacy: anyone who already knows a
recipient id and the salt could recompute its token, so tokens are treated as
sensitive, expire quickly, and are never logged alongside raw identifiers.
A signal is a risk warning for investigation, never a guilt verdict.
"""
import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from . import database as db

CATEGORIES = ("mule_recipient", "mismatch_pattern", "velocity_anomaly")
DEFAULT_TTL_HOURS = 24
_TOKEN_SALT_FILE = "signal_salt"


def _salt():
    from .config import DATA_DIR
    p = DATA_DIR / ".signal_salt"
    if p.exists():
        return bytes.fromhex(p.read_text().strip())
    s = secrets.token_bytes(32)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(s.hex())
    except OSError:
        pass
    return s


def entity_token(recipient_id):
    """Pseudonymous cross-bank token. Input validation: non-empty string."""
    if not isinstance(recipient_id, str) or not recipient_id.strip():
        raise ValueError("recipient_id must be a non-empty string")
    return hmac.new(_salt(), recipient_id.strip().encode(),
                    hashlib.sha256).hexdigest()[:32]


@dataclass
class Signal:
    id: str
    source_bank: int
    created_at: str
    expires_at: str
    category: str
    risk_score: float
    model_version: str
    entity_token: str
    provenance: str
    status: str


def init_alert_tables():
    from .config import DB_PATH
    c = sqlite3.connect(str(DB_PATH), timeout=30)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS signals(id TEXT PRIMARY KEY, source_bank INTEGER,
      created_at TEXT, expires_at TEXT, category TEXT, risk_score REAL,
      model_version TEXT, entity_token TEXT, provenance TEXT, status TEXT);
    CREATE INDEX IF NOT EXISTS idx_signals_token ON signals(entity_token, status);
    CREATE TABLE IF NOT EXISTS signal_events(id INTEGER PRIMARY KEY, signal_id TEXT,
      kind TEXT, detail TEXT, user_id TEXT, created_at TEXT);
    """)
    c.commit(); c.close()


def _now():
    return datetime.now(timezone.utc)


def publish(source_bank, category, risk_score, token, model_version,
            provenance, user, ttl_hours=DEFAULT_TTL_HOURS):
    """Publish a risk signal. Requires an authenticated user (any role)."""
    if user is None:
        raise PermissionError("Authentication required.")
    if source_bank not in range(5):
        raise ValueError("source_bank must be 0..4")
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    if not isinstance(risk_score, (int, float)) or not (0 <= risk_score <= 1):
        raise ValueError("risk_score must be 0..1")
    if not isinstance(token, str) or len(token) != 32:
        raise ValueError("entity token must be a 32-char pseudonym")
    if not model_version or len(str(model_version)) > 64:
        raise ValueError("invalid model_version")
    init_alert_tables()
    t = _now()
    exp = t + timedelta(hours=ttl_hours)
    from .config import DB_PATH
    c = sqlite3.connect(str(DB_PATH), timeout=30)
    # Deduplication: same token+category from same bank still active -> reuse
    r = c.execute("SELECT id FROM signals WHERE entity_token=? AND category=? "
                  "AND source_bank=? AND status='active' AND expires_at>?",
                  (token, category, source_bank, t.isoformat())).fetchone()
    if r:
        c.close()
        return r[0], False
    sid = "sig_" + secrets.token_hex(8)
    c.execute("INSERT INTO signals VALUES(?,?,?,?,?,?,?,?,?,?)",
              (sid, source_bank, t.isoformat(), exp.isoformat(), category,
               float(risk_score), str(model_version), token, str(provenance)[:500],
               "active"))
    c.execute("INSERT INTO signal_events(signal_id,kind,detail,user_id,created_at)"
              " VALUES(?,?,?,?,?)",
              (sid, "published", f"bank {source_bank} {category} score {risk_score}",
               str(user.get("name")), t.isoformat()))
    c.commit(); c.close()
    db.audit(str(user.get("name")), "signal_publish", f"{sid} bank {source_bank}")
    return sid, True


def active_signals(bank_id=None, token=None):
    """Signals visible to a receiving bank: active, unexpired, from OTHER banks."""
    init_alert_tables()
    from .config import DB_PATH
    t = _now().isoformat()
    q = ("SELECT * FROM signals WHERE status='active' AND expires_at>?")
    args = [t]
    if bank_id is not None:
        q += " AND source_bank!=?"
        args.append(bank_id)
    if token is not None:
        q += " AND entity_token=?"
        args.append(token)
    q += " ORDER BY created_at DESC"
    c = sqlite3.connect(str(DB_PATH), timeout=30)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(q, args).fetchall()]
    c.close()
    return rows


def expire_due():
    init_alert_tables()
    from .config import DB_PATH
    t = _now().isoformat()
    c = sqlite3.connect(str(DB_PATH), timeout=30)
    due = [r[0] for r in c.execute("SELECT id FROM signals WHERE status='active'"
                                   " AND expires_at<=?", (t,)).fetchall()]
    for sid in due:
        c.execute("UPDATE signals SET status='expired' WHERE id=?", (sid,))
        c.execute("INSERT INTO signal_events(signal_id,kind,detail,user_id,created_at)"
                  " VALUES(?,?,?,?,?)", (sid, "expired", "ttl elapsed", "system", t))
    c.commit(); c.close()
    for sid in due:
        db.audit("system", "signal_expire", sid)
    return len(due)


def match(token, receiving_bank, user):
    """Match an incoming transaction's token against active signals from other
    banks. Returns an explainable risk warning, never a guilt verdict."""
    if user is None:
        raise PermissionError("Authentication required.")
    if not isinstance(token, str) or not token:
        raise ValueError("entity token required")
    t0 = _now()
    expire_due()
    hits = active_signals(bank_id=receiving_bank, token=token)
    latency_ms = (_now() - t0).total_seconds() * 1000
    from .config import DB_PATH
    c = sqlite3.connect(str(DB_PATH), timeout=30)
    for h in hits:
        c.execute("INSERT INTO signal_events(signal_id,kind,detail,user_id,created_at)"
                  " VALUES(?,?,?,?,?)",
                  (h["id"], "matched",
                   f"bank {receiving_bank} token match", str(user.get("name")),
                   _now().isoformat()))
    c.commit(); c.close()
    if user:
        db.audit(str(user.get("name")), "signal_match",
                 f"bank {receiving_bank} hits {len(hits)}")
    out = []
    for h in hits:
        age_h = (_now() - datetime.fromisoformat(h["created_at"])).total_seconds() / 3600
        out.append({"signal_id": h["id"], "source_bank": h["source_bank"],
                    "category": h["category"], "risk_score": h["risk_score"],
                    "model_version": h["model_version"],
                    "age_hours": round(age_h, 2), "expires_at": h["expires_at"],
                    "reason": (f"Pseudonymous token seen in fraud-linked activity at bank "
                               f"{h['source_bank']} ({h['category']}, score {h['risk_score']}). "
                               "Risk warning for verification, not a guilt finding."),
                    "latency_ms": round(latency_ms, 2)})
    return out
