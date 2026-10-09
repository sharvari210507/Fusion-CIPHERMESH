"""Dataset download, Parquet cache, bank partitions, statistics."""
import json
import pandas as pd
from pathlib import Path
from .config import DATA_DIR, DATASET_ID, LABEL, BANK_COL, N_BANKS

EXPECTED_COLS = {"step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                 "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
                 "isFlaggedFraud", "BankID"}


def paths():
    return DATA_DIR / "train.parquet", DATA_DIR / "test.parquet", DATA_DIR / "stats.json"


def is_cached():
    tr, te, _ = paths()
    return tr.exists() and te.exists()


def download_and_cache(progress=None):
    from datasets import load_dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset(DATASET_ID)
    except Exception as e:
        raise RuntimeError(f"Dataset download failed for {DATASET_ID}: {e}")
    names = list(ds.keys())
    if "train" not in names or "test" not in names:
        raise RuntimeError(f"Unexpected splits {names}; need train/test.")
    for split in ("train", "test"):
        df = ds[split].to_pandas()
        cols = set(df.columns)
        if cols != EXPECTED_COLS:
            raise RuntimeError(f"Schema mismatch in {split}: got {sorted(cols)}")
        for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
            df[c] = df[c].astype("float32")
        for c in ["step", "isFraud", "isFlaggedFraud", "BankID"]:
            df[c] = df[c].astype("int32")
        df["type"] = df["type"].astype("category")
        df.to_parquet(DATA_DIR / f"{split}.parquet", index=False)
        if progress:
            progress(split)
    compute_stats()
    return True


def load_split(split="train", columns=None):
    tr, te, _ = paths()
    p = tr if split == "train" else te
    if not p.exists():
        raise RuntimeError("Dataset not cached. Restart the app with internet once.")
    return pd.read_parquet(p, columns=columns)


def bank_partition(split, bank_id):
    df = load_split(split)
    return df[df[BANK_COL] == bank_id].reset_index(drop=True)


def compute_stats():
    tr, te, sp = paths()
    train = pd.read_parquet(tr, columns=["BankID", "isFraud", "amount", "type"])
    test = pd.read_parquet(te, columns=["BankID", "isFraud"])
    stats = {"n_train": int(len(train)), "n_test": int(len(test)),
             "fraud_rate_train": float(train["isFraud"].mean()),
             "banks": {}, "type_mix": train["type"].value_counts().to_dict()}
    for b in range(N_BANKS):
        d = train[train["BankID"] == b]
        dt = test[test["BankID"] == b]
        stats["banks"][str(b)] = dict(
            train_rows=int(len(d)), test_rows=int(len(dt)),
            fraud_train=int(d["isFraud"].sum()), fraud_test=int(dt["isFraud"].sum()),
            fraud_rate=float(d["isFraud"].mean()) if len(d) else 0.0,
            mean_amount=float(d["amount"].mean()) if len(d) else 0.0)
    sp.write_text(json.dumps(stats, indent=1))
    return stats


def get_stats():
    _, _, sp = paths()
    if sp.exists():
        return json.loads(sp.read_text())
    if is_cached():
        return compute_stats()
    return {"status": "not_cached"}
