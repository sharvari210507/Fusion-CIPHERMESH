# Experiment Methodology Audit — CIPHERMESH/FedGuard (CSB-04)

Date: 2026-10-10. Branch: `fix/ui-visible-defects`. Audit-only: no production
training or evaluation code was changed, no records rewritten, no experiments
rerun. Evidence is implementation + stored artifacts + focused read-only checks.

## Executive verdict: VALID WITH CAVEATS

The core federated loop is sound (disjoint publisher splits, stateless
features, validation-tuned threshold, test-only reporting, imbalance-aware
metrics), and the live dashboard numbers for the active model (job 3) are
valid. The E1–E6 comparison suite and the stored job-2 recall column are NOT
valid for like-for-like comparison without the corrections listed below.

## Provenance (checked, not assumed)

- Training jobs in `data/fedguard.db`: 1, 2, 3 completed (configs below);
  `active_model='3'`. `models/global_run_1..3.joblib` and
  `results/run_1..3.json` (+ `_perbank`) exist on disk.
- `git log -- backend/metrics.py`: `baf1391` (2026-10-09 14:29:05 +0530 =
  08:59:05 UTC) fixed `recall_at_fpr` ascending→descending sort and flipped
  `DEFAULT_RUN` clip `None`→`1.0`. Job finished_at (UTC): 1→08:51:40,
  2→08:55:22, 3→08:59:58. So jobs 1–2 ran pre-fix code; job 3 ran post-fix.
- `results/experiments.json` config: `seeds:[42]`, extra keys `fed_clip`,
  `note`; its `E6_noise` carries `clip_norm_used` + `note`, keys the current
  `run_suite` never writes (it writes only `clip_norm_measured` + `sweep`).
  The artifact is stale relative to current code and single-seed while
  `EXP_DEFAULT` (`backend/config.py:12`) specifies 3 seeds.

## Findings

### F1 [High] Job-2 stored recall@1%FPR is wrong; job-3 is correct
- Evidence: `backend/metrics.py:12-20` (current, descending sort) vs
  `git show baf1391 -- backend/metrics.py` (`np.argsort(s)` ascending).
  DB: job 2 recall is exactly `0.0` in all 5 rounds with PR-AUC 0.747–0.765;
  job 3 (identical weights/PR-AUC) has recall 0.874–0.916.
  Read-only check: on separable synthetic labels the old ordering yields
  recall 0.0 where the current implementation yields 1.0 (verified at audit).
- Status: confirmed defect in stored data, code already fixed.
- Demo rule: never cite job-2 recall; cite job 3 (0.906 last round).
  Historical display is annotated separately (Control Room `run_status`).

### F2 [High] E3 "pooled reference" is not a fair comparator
- Evidence: `backend/experiments.py:103-115` trains pooled in ONE
  `train_round` call with `local_epochs=rounds*epochs` (single SGD fit),
  versus `_quick_train` (`experiments.py:30-36`) doing `rounds` sequential
  FedAvg aggregations. Result: E3 PR-AUC 0.085 vs E2 0.75 — an optimization
  artifact, not evidence federated beats centralized.
- Status: confirmed methodological defect. Safe claim: "E3 exists as an
  unlike-for-like reference"; unsafe: any pooled-vs-federated comparison.

### F3 [High] E-suite is single-seed; every std is 0.0
- Evidence: artifact `config.seeds=[42]` vs `EXP_DEFAULT.seeds=[42,43,44]`;
  all `std: 0.0`. E5 additionally trains the global arm on `seeds[0]` only
  while averaging locals over seeds (`experiments.py:161-178`), mixing seed
  effects into `delta`.
- Status: confirmed. Directional gaps (E2 1→5 banks +0.04, E5 deltas
  +0.48..+0.77) are large enough to be suggestive but not publishable.

### F4 [Medium] Experiment F1 (thr 0.5) vs dashboard F1 (thr ~0.88) incomparable
- Evidence: `experiments.py:54` `_eval(w, mode, thr=0.5)` vs
  `backend/jobs.py:137-142` (threshold from `M.best_threshold` on pooled
  client val, then `full_metrics(yte, s, thr)`). PR-AUC is threshold-free and
  comparable; F1/recall are not.
- Status: confirmed. Compare PR-AUC across E-tables and runs, never F1.

### F5 [Medium] Engineered error-balance features trivialize the task
- Evidence: `backend/features.py:26-27` (`err_orig`, `err_dest` from balance
  arithmetic); Data Explorer itself warns (`views/data_explorer.py:31-32`),
  Threat Model discloses leakage (`views/threat_model.py` BODY).
  Default run mode is `raw` (`backend/config.py:10`), which is what job 3
  used. No learned preprocessing exists at all: `transform`
  (`features.py:15-29`) is stateless — verified batch/order-invariant with
  no fit/scaler/encoder ops.
- Status: confirmed risk, contained by default. Demo in `raw`; do not claim
  engineered-mode generality.

### F6 [Medium] Scoring demo loads real TEST rows into a test-evaluated model
- Evidence: `backend/scoring.py:64-72` (`example_row` reads the test split);
  `views/score_transaction.py:30-36`. Model was evaluated on test
  (`jobs.py:136,141`). Scoring a test row with it is demo-only leakage.
- Status: confirmed (demo scope). Safe: label as illustrative; fix later by
  sourcing examples from train or synthetic input.

### F7 [Medium] Per-bank model quality exists but is not on the dashboard
- Evidence: `results/run_{jid}_perbank.json` written (`jobs.py:150-156`);
  Overview shows per-bank DATASET stats (`views/overview.py` via
  `data.compute_stats`) but only aggregate model metrics. Non-IID spread is
  real (train fraud rate 0.00091 bank 0 → 0.00175 bank 4; E5 deltas vary),
  so aggregates could hide a failing bank in the UI even though artifacts
  exist.
- Status: confirmed presentation gap, not a metric defect.

### F8 [Low] Test reporting every round, no early stopping on test
- Evidence: `jobs.py:128-143` loops fixed `rounds`, evaluates test per round
  for reporting; threshold comes from pooled client val
  (`jobs.py:124-140`, val drawn from train via
  `federated.py:50-56` 90/10 split). No stopping/selection on test.
- Status: confirmed sound. Residual risk is only if someone cherry-picks the
  best round; dashboard shows the last round.

### F9 [Low] Correct clipping target and units; metadata is counts, not rows
- Evidence: `federated.py:84-88` clips the weight DELTA
  (`local_w - global_w`) via `privacy.clip_update` (`privacy.py:4-11`, L2
  norm of the coefficient delta); `validate_message` (`federated.py:27-39`)
  rejects DataFrames/non-numeric/non-finite; `database.add_message`
  (`database.py:103-125`) persists `update.nbytes`, `n_samples`,
  norms, `rows_transmitted=0` with `payload_type` evidence.
- Status: confirmed — "clipping" clips what is claimed, in L2 weight-delta
  units. `n_samples`/`update_bytes` crossing the boundary are aggregates.

### F10 [Low] Train/test construction and bank separation are honest
- Evidence: publisher-provided splits required by schema gate
  (`data.py:29-35`, exact column match); bank partitions are disjoint
  `BankID==b` filters (`data.py:56-58`); `BankID`, labels and
  `isFlaggedFraud` are NOT features (verified at runtime:
  `FEATURES_RAW`/`FEATURES_ENG` contain only log balances, hour, type
  one-hots, error terms).
- Caveat (needs evidence): full-row disjointness of publisher splits was not
  re-verified (5.7M-row join deemed out of scope); account IDs legitimately
  recur across splits as distinct transactions.

### F11 — Privacy and alert claims: what actually crosses the boundary
- Training: only `UpdateMessage` (numeric array + `n_samples`, norms,
  `noise_std`) — substantiated (`federated.py`, `database.py` above).
  Audit log holds usernames/job IDs/scores; `predictions` stores
  user-supplied input JSON, not dataset rows. No formal DP or secure
  aggregation — explicitly disclaimed (`views/privacy_audit.py:32-33`,
  Threat Model). Safe: "raw rows never leave the bank (prototype)";
  unsafe: "differentially private" / "anonymous".
- Alerts: token-only `HMAC(recipient, server salt)` (`alerts.py:42-47`,
  salt file `data/.signal_salt`); `publish` validates and dedups
  (`alerts.py:82-120`); `match` is token equality against unexpired
  other-bank signals (`alerts.py:161-195`). Anyone holding identifier+salt
  could recompute a token — documented in code and UI captions. Safe as
  "pseudonymous investigation warnings"; unsafe as "unlinkable".
- Two-hour replay: REAL linkage confirmed read-only (16 Bank0-fraud →
  Bank1-shared-recipient candidates in test split, e.g. `C417640852`
  step 43); real active-model scoring; SIMULATED clock honestly labeled in
  `views/cross_bank_alerts.py:15-18` and trail fields. Safe: "simulated-clock
  replay on real test records"; unsafe: "live production detection".

## Q-by-Q answers (1–11)

1. Preprocessing: no learned encoders/scalers/selectors exist — separation
   holds trivially (F5 evidence).
2. Test labels: never used for training/aggregation/threshold/early-stopping
   (threshold from train-derived pooled val, F8); test used for per-round
   reporting only. Scoring demo reuses test rows illustratively (F6).
3. Same test records/features/eval code for local vs federated: yes,
   `_eval` on the shared test split with identical mode (`experiments.py`);
   seeds asymmetric in E5 (F3); F1 thresholds differ between E-suite (0.5)
   and runs (~0.88) (F4).
4. Imbalance: PR-AUC (AP), recall@1%FPR with correct descending ranking,
   precision/recall/F1 + confusion persisted per round; operating threshold
   via best-F1 on val PR curve, not test (F8, `metrics.py:23-39`).
5. Banks: disjoint partitions, no BankID feature, no duplicated training
   records (F10).
6. Non-IID: yes (fraud-rate ~2x spread); per-bank dataset stats on
   dashboard, per-bank model metrics only in artifacts (F7).
7. Clipping: verified on the delta, L2 units (F9).
8. Saved records vs code: run artifacts match code; `experiments.json` is
   stale vs current code (extra `fed_clip`/`note`/`clip_norm_used` keys,
   1 seed) — dashboard reads persisted files, no hard-coded metrics (F3,
   `views/experiments.py:17-24`).
9. Fairness: federated-vs-local directionally fair on shared test but
   single-seed (F3); pooled E3 unfair (F2); F1 cross-comparison invalid (F4).
10. Privacy boundary: substantiated as prototype, no formal guarantees (F11).
11. Alert matching: real dataset linkage + real scoring, simulated clock
    disclosed (F11).

## Tests actually run (all non-destructive, working tree clean after)

- `pytest tests/test_backend.py tests/test_correctness.py tests/test_fedavg.py
  tests/test_update_protection.py tests/test_trust_ledger.py
  tests/test_privacy_audit.py -q` → **24 passed**.
- Read-only inline checks: Bank0→Bank1 linkage (16 candidates); feature
  statelessness (batch/order-invariant, no fit ops); artifact-vs-code key
  mismatch; old-vs-new recall on synthetic labels (0.0 vs 1.0); live
  annotation of real DB jobs (1→unstable, 2→outdated recall, 3→active).
- No full-dataset training or experiment reruns performed.

## Safe vs unsafe demo claims

Safe: job-3 PR-AUC ~0.755 / recall@1%FPR ~0.91 (last round, val-tuned
threshold ~0.88); federation helps directionally (E2/E5 PR-AUC gaps);
token-only alerts with simulated clock; raw rows never leave banks
(prototype, no formal DP). Unsafe: job-2 recall values; any E3-based
centralized comparison; single-seed stds; engineered-mode performance;
"real-time detection"; "differential privacy"; per-bank health from the
dashboard alone.

## Minimum corrective actions (priority order)

1. Keep the Task-4 historical annotations (job-2 recall must stay flagged;
   no backfill possible — per-round scores were not stored).
2. Rerun E-suite with `EXP_DEFAULT` 3 seeds and epoch-parity E3 before
   citing comparison numbers (needs a 30–60 min compute window; NOT done
   in this task).
3. Surface per-bank test metrics (`run_*_perbank.json`) on the dashboard
   with the active-model threshold noted.
4. Source Score-page examples from train/synthetic, not the test split.
5. Align experiment F1 threshold with the run methodology or label the
   0.5-vs-tuned difference in experiment captions.

## E1–E6 rerun verdict

Rerun required for any citable comparison (reasons F2–F4 plus stale artifact
keys); not required to support the architectural demo, which rests on job-3
run metrics and the live replay. The rerun is a separate implementation task
with compute cost, explicitly out of scope here.
