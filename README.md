# FedGuard (Fusion-CIPHERMESH) — CSB-04

Privacy-preserving fraud signal sharing across banks: five simulated banks train one shared
fraud-detection model with federated averaging. Only numeric weight updates leave each bank.
Prototype for the SWIFT CSB-04 hackathon problem statement. Synthetic data only.

## Architecture (one paragraph)

Cached Parquet splits (`data/`) are partitioned by `BankID` into five `BankClient`s holding
only their own rows. Each round, clients train a logistic-regression (sklearn `SGDClassifier`,
warm-started from global weights via `coef_init`/`intercept_init`) on local features, clip/add
Gaussian noise optionally, and return an `UpdateMessage` (numeric array + counts only). The
aggregator validates messages (rejects DataFrames, NaN, bad shapes) and applies sample-weighted
FedAvg. A background job manager (`backend/jobs.py`) runs training/experiments, writing round
metrics and message records to SQLite (`data/fedguard.db`) plus `models/`/`results/`. Streamlit
pages call only `backend/` services; roles are enforced in the backend.

## Setup (Kali/Ubuntu/Windows)

Requires Python 3.10–3.12 per spec. **Deviation:** this laptop only has Python 3.13/3.14
available and no compliant interpreter could be installed from a trusted source, so the project
`.venv` uses Python 3.13. All pinned packages and tests pass on it.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## First run (dataset download + offline cache)

```bash
streamlit run app.py
```

First start needs internet once: downloads `flwrlabs/fed-fraud-paysim-banks` (~6.36M rows)
via `datasets` and caches `data/train.parquet`, `data/test.parquet`, `data/stats.json`.
Later starts work offline. A default 5-bank run auto-starts when no completed run exists
(`AUTO_START_JOBS` in `backend/config.py`). Never commit `data/`, `models/`, `results/`, `*.db`.

## Google sign-in setup (optional) / local fallback

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, create an OAuth client in
Google Cloud Console, add `http://localhost:8501/oauth2callback` as redirect URI, add test users
while the consent screen is in testing mode, fill `client_id`/`client_secret`/`cookie_secret` and
`[app] admin_users`. When unconfigured, the button is hidden and local registration/login works
offline (username 3–32 chars, password 10+ chars, bcrypt hash, lockout after 5 failures).
No seeded accounts. Admin rights come only from `admin_users`. Local sessions live in
`st.session_state` and end on tab reload (known limitation).

## Run / tests

```bash
streamlit run app.py        # open http://localhost:8501
pytest tests/ -q            # 8 backend tests, synthetic data only
```

## Pages

Monitor: Overview, Data Explorer. Federation: Control Room (admin starts/cancels runs, live
PR-AUC/recall chart, set active model), Experiments E1–E6 with generated captions.
Security: Privacy Audit (per-message bytes/samples/rows_transmitted=0 evidence, noise note),
Threat Model. Tools: Score Transaction (validated inputs, global probability vs threshold, top
feature contributions, real fraud/non-fraud examples from test split). Administration (admin only).

## Team handoff

- Independent of the full dataset (small synthetic data): UI accessibility review, threat-model
  text (`views/threat_model.py`), Times New Roman/icon styling (`ui/theme.py`), new edge-case
  tests in `tests/test_backend.py`, README/acceptance checklist updates, experiment-caption
  wording (`views/experiments.py`).
- Needs the main laptop + cached dataset: default training runs, E1–E6 suite, Data Explorer
  numbers, scoring examples, performance benchmark.
- Workflow: `git fetch origin && git checkout -b feat/<name>` from latest `main`; focused
  conventional commits (`feat:`, `test:`, `fix:`, `docs:`); open a PR, never push to `main`
  directly; contract: pages call only `backend/` functions returning plain data; reviewers check
  no raw DataFrames cross into the aggregator, no invented numbers, SQL parameterized.
- Module owners: data/loader, features+metrics, federated+privacy, jobs+experiments, auth+db,
  views+theme, tests+README (assign names per laptop).

## Privacy model, threat model, limitations

Noise = Gaussian on clipped updates; **no formal epsilon/delta guarantee** (no accounting).
FedAvg alone does not prevent update leakage; secure aggregation/TLS/robust aggregation are
future work. All data is synthetic PaySim mobile-money; banks simulated on one machine; balance
columns leak labels (raw mode default, engineered mode shown for comparison). Accuracy is never
a headline metric (PR-AUC, recall@1%FPR, F1). Pooled E3 is a reference requiring raw-data sharing.

## Attribution

Dataset: `flwrlabs/fed-fraud-paysim-banks`, Hugging Face, license CC-BY-4.0. Same credit in app footer.

## Implementation checklist

- [x] Slice 1: env, skeleton, config, dataset cache + stats
- [x] Slice 2: features, local model, FedAvg, clipping/noise, metrics
- [x] Slice 3: SQLite, background jobs, progress/cancel/restart
- [x] Slice 4: local auth + optional Google, roles, audit log
- [x] Slice 5: app shell, theme, login, Overview, Control Room
- [x] Slice 6: Data Explorer, E1–E6, Privacy Audit, Score, Threat Model, Admin
- [x] Slice 7: tests (8 pass), README, security review
- [ ] Full 6.3M-row download + measured default-run benchmark (in progress on first boot)
- [ ] Traceability table sign-off after live training numbers observed

Known deviations: (1) Python 3.13 used (3.10–3.12 unavailable); (2) `architecture.txt` (simple
no-auth/no-DB prototype) is superseded by the Build Specification, which is authoritative;
only one spec MD file existed on Desktop (no `(1)` variant), so no merge was needed.
