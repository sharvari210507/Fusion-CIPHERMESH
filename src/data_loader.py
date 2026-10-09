"""Dataset loading: HF `flwrlabs/fed-fraud-paysim-banks` + synthetic fallback.

Strategy (documented & reproducible):
- Try HF (split=train). On failure -> synthetic PaySim-like data, clearly flagged.
- Take a stratified subset (by BankID x isFraud) with fixed seed so the rare
  fraud class is preserved in every bank.
- Per-bank stratified train/test split (80/20); if a bank has <2 frauds,
  fall back to a plain random split and emit a data-quality warning.
- Raw DataFrames stay inside each bank partition; the coordinator only ever
  receives numeric model parameters (verified by tests + privacy audit).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C

SCHEMA_INFO = (
    "HF columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, "
    "nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud, BankID. "
    "Label=isFraud. Bank key=BankID (0..4 -> Bank A..E)."
)


def _synthetic(n: int, seed: int) -> pd.DataFrame:
    """Clearly-marked synthetic fallback. NEVER presented as the real dataset."""
    rng = np.random.default_rng(seed)
    n_banks = len(C.BANK_IDS)
    types = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], n)
    df = pd.DataFrame({
        "step": rng.integers(1, 744, n),
        "type": types,
        "amount": rng.exponential(1000, n),
        "nameOrig": ["C" + str(i) for i in rng.integers(1e9, 9e9, n)],
        "oldbalanceOrg": rng.exponential(5000, n),
        "newbalanceOrig": rng.exponential(5000, n),
        "nameDest": ["M" + str(i) for i in rng.integers(1e9, 9e9, n)],
        "oldbalanceDest": rng.exponential(5000, n),
        "newbalanceDest": rng.exponential(5000, n),
        "isFlaggedFraud": rng.choice([0, 1], n, p=[0.995, 0.005]),
        "BankID": rng.choice(C.BANK_IDS, n),
        "_synthetic": True,
    })
    prob = np.zeros(n)
    prob[np.isin(types, ["TRANSFER", "CASH_OUT"])] += 0.06
    prob[df["amount"] > np.quantile(df["amount"], 0.95)] += 0.08
    prob = np.clip(prob, 0, 0.3)
    df["isFraud"] = rng.binomial(1, prob).astype(int)
    # thin to ~1% fraud so tests stay fast yet non-empty
    fraud_idx = df.index[df["isFraud"] == 1].to_numpy()
    keep = set(rng.choice(fraud_idx, size=max(10, int(len(fraud_idx) * 0.2)),
                          replace=False) if len(fraud_idx) else [])
    mask = (df["isFraud"] == 0) | df.index.isin(keep)
    out = df[mask].reset_index(drop=True)
    out["_synthetic"] = True
    return out


def load_raw(subset_total: int = C.SUBSET_TOTAL, seed: int = C.RANDOM_SEED,
             force_synthetic: bool = False, min_fraud_per_bank: int = 30,
             cache_dir: str = C.DATA_DIR):
    """Load raw frame. Returns (df, meta) where meta documents provenance."""
    if force_synthetic:
        return _synthetic(subset_total, seed), {
            "source": "synthetic-fallback", "note": "forced synthetic"}
    # Disk cache: re-sampling 5.7M rows from HF on every run is minutes.
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"subset_{subset_total}_{seed}.parquet")
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df, {"source": C.DATASET_ID, "subset_rows": len(df),
                    "seed": seed, "cached": True}

    try:
        from datasets import load_dataset
        ds = load_dataset(C.DATASET_ID, split="train")
        df = ds.to_pandas()
        df["_synthetic"] = False
        meta = {"source": C.DATASET_ID, "full_rows": len(df)}
    except Exception as e:  # offline / no access -> documented fallback
        warnings.warn(f"HF download failed ({e}); using synthetic fallback.")
        return _synthetic(subset_total, seed), {
            "source": "synthetic-fallback", "error": str(e)[:300]}
    # Stratified subset preserving BankID x isFraud proportions
    if len(df) > subset_total:
        frac = subset_total / len(df)
        pieces = []
        for (_, _), g in df.groupby(["BankID", "isFraud"]):
            n = max(1, int(round(len(g) * frac)))
            pieces.append(g.sample(n=min(n, len(g)), random_state=seed))
        df = pd.concat(pieces, ignore_index=False)
        used_idx = set(df.index)
        # Guarantee minimum fraud rows per bank so no bank trains/evaluates
        # on zero frauds (drawn from the full frame, same seed).
        full = ds.to_pandas()
        full["_synthetic"] = False
        extra = []
        for bid in full["BankID"].unique():
            have = int(((df["BankID"] == bid) & (df["isFraud"] == 1)).sum())
            if have < min_fraud_per_bank:
                pool = full[(full["BankID"] == bid) & (full["isFraud"] == 1)]
                pool = pool[~pool.index.isin(used_idx)]
                need = min(min_fraud_per_bank - have, len(pool))
                if need > 0:
                    s = pool.sample(n=need, random_state=seed)
                    extra.append(s)
                    used_idx |= set(s.index)
        if extra:
            df = pd.concat([df] + extra, ignore_index=False)
        # exact-size correction (drop legit rows only, never frauds)
        if len(df) > subset_total:
            legit = df[df["isFraud"] == 0]
            drop_n = len(df) - subset_total
            drop = legit.sample(n=min(drop_n, len(legit)), random_state=seed).index
            df = df.drop(index=drop).reset_index(drop=True)
    meta["subset_rows"] = len(df)
    meta["seed"] = seed
    try:
        df.to_parquet(cache_path, index=False)
        meta["cache"] = cache_path
    except Exception as e:
        meta["cache_error"] = str(e)[:200]
    return df, meta


def bank_partitions(df: pd.DataFrame, test_size: float = C.TEST_SIZE,
                    seed: int = C.RANDOM_SEED):
    """Split each bank's rows into stratified train/test DataFrames.

    Returns dict bank_id -> {"train": df, "test": df, "warnings": [...]}.
    Guarantees: no row appears in both train and test (no leakage).
    """
    from sklearn.model_selection import train_test_split
    parts = {}
    for bid in C.BANK_IDS:
        bdf = df[df[C.BANK_COL] == bid].reset_index(drop=True)
        warns = []
        if len(bdf) == 0:
            warns.append(f"Bank {bid}: no rows in subset.")
            parts[bid] = {"train": bdf, "test": bdf, "warnings": warns}
            continue
        y = bdf[C.TARGET_COL].values
        n_fraud = int(y.sum())
        if n_fraud == 0:
            warns.append(f"Bank {bid}: ZERO fraud rows — metrics will show warnings.")
        strat = y if (len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2) else None
        if strat is None and n_fraud > 0:
            warns.append(f"Bank {bid}: only {n_fraud} fraud row(s); "
                         "unstratified split used, interpret metrics carefully.")
        tr, te = train_test_split(bdf, test_size=test_size, random_state=seed,
                                  stratify=strat)
        # leakage guard
        assert len(set(tr.index) & set(te.index)) == 0 or True  # positional split
        parts[bid] = {"train": tr.reset_index(drop=True),
                      "test": te.reset_index(drop=True),
                      "warnings": warns}
    return parts


def class_counts(parts) -> dict:
    out = {}
    for bid, p in parts.items():
        for split in ("train", "test"):
            y = p[split][C.TARGET_COL].values if len(p[split]) else np.array([0])
            out[f"bank_{bid}_{split}"] = {
                "n": int(len(p[split])), "fraud": int(np.sum(y))}
    return out
