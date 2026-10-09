"""SQLite schema + parameterized queries."""
import json
import sqlite3
from datetime import datetime, timezone
from .config import DB_PATH


def _c():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    c = _c()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE,
      email TEXT, display_name TEXT, provider TEXT, password_hash TEXT, role TEXT DEFAULT 'analyst',
      created_at TEXT, last_login_at TEXT, failed_attempts INTEGER DEFAULT 0, locked_until TEXT);
    CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY, kind TEXT, status TEXT,
      config_json TEXT, requested_by TEXT, created_at TEXT, started_at TEXT, finished_at TEXT,
      error TEXT, result_path TEXT);
    CREATE TABLE IF NOT EXISTS round_metrics(job_id INTEGER, round INTEGER, pr_auc REAL,
      recall_at_1pct_fpr REAL, f1 REAL, created_at TEXT);
    CREATE TABLE IF NOT EXISTS messages(job_id INTEGER, round INTEGER, bank_id INTEGER,
      n_samples INTEGER, update_bytes INTEGER, rows_transmitted INTEGER,
      norm_before_clip REAL, norm_after_clip REAL, noise_std REAL, created_at TEXT);
    CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, user_id TEXT, input_json TEXT,
      job_id INTEGER, bank_id INTEGER, p_global REAL, p_local REAL, created_at TEXT);
    CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY, user_id TEXT, action TEXT,
      detail TEXT, created_at TEXT);
    CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
    """)
    c.commit(); c.close()


def now():
    return datetime.now(timezone.utc).isoformat()


def audit(user_id, action, detail=""):
    c = _c()
    c.execute("INSERT INTO audit_log(user_id,action,detail,created_at) VALUES(?,?,?,?)",
              (str(user_id), str(action)[:100], str(detail)[:2000], now()))
    c.commit(); c.close()


def create_job(kind, config, requested_by):
    c = _c()
    cur = c.execute("INSERT INTO jobs(kind,status,config_json,requested_by,created_at) VALUES(?,?,?,?,?)",
                    (kind, "queued", json.dumps(config), str(requested_by), now()))
    jid = cur.lastrowid
    c.commit(); c.close()
    return jid


def set_job(jid, **kw):
    c = _c()
    for k, v in kw.items():
        if k in ("status", "started_at", "finished_at", "error", "result_path"):
            c.execute(f"UPDATE jobs SET {k}=? WHERE id=?", (v, jid))
    c.commit(); c.close()


def get_job(jid):
    c = _c(); c.row_factory = sqlite3.Row
    r = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    c.close()
    return dict(r) if r else None


def list_jobs(limit=50):
    c = _c(); c.row_factory = sqlite3.Row
    rows = c.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def add_round_metric(jid, rnd, pr, rec, f1):
    c = _c()
    c.execute("INSERT INTO round_metrics VALUES(?,?,?,?,?,?)", (jid, rnd, pr, rec, f1, now()))
    c.commit(); c.close()


def round_metrics(jid):
    c = _c(); c.row_factory = sqlite3.Row
    rows = c.execute("SELECT * FROM round_metrics WHERE job_id=? ORDER BY round", (jid,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def add_message(jid, rnd, msg):
    c = _c()
    c.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?)",
              (jid, rnd, msg.bank_id, msg.n_samples, int(msg.update.nbytes), 0,
               msg.norm_before_clip, msg.norm_after_clip, msg.noise_std, now()))
    c.commit(); c.close()


def get_messages(jid):
    c = _c(); c.row_factory = sqlite3.Row
    rows = c.execute("SELECT * FROM messages WHERE job_id=? ORDER BY round,bank_id", (jid,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def set_setting(k, v):
    c = _c()
    c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (k, str(v)))
    c.commit(); c.close()


def get_setting(k):
    c = _c()
    r = c.execute("SELECT value FROM settings WHERE key=?", (k,)).fetchone()
    c.close()
    return r[0] if r else None


def query(sql, args=(), limit=500):
    allowed = ("users", "audit_log", "jobs", "predictions")
    if not any(t in sql for t in allowed) or not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT on known tables.")
    c = _c(); c.row_factory = sqlite3.Row
    rows = c.execute(sql, args).fetchmany(limit)
    c.close()
    return [dict(r) for r in rows]
