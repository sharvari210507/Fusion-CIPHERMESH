"""Consistent preprocessing shared by every bank.

Rules enforced here (see spec section 5):
- Target (isFraud), bank id (BankID) and identifiers (nameOrig/nameDest)
  are NEVER model inputs.
- Categorical `type` is one-hot encoded with a FIXED category order so every
  bank produces identical columns even if a category is missing locally.
- StandardScaler is fit on TRAINING data only, then applied to val/test.
- Feature order is fixed and saved; validated before training/aggregation.

Potential-leakage note: `isFlaggedFraud` is kept as a feature but it is an
output of a legacy rule engine (only 14 positives / 5.7M rows). It is
near-constant and must not be trusted as a real-time signal. Documented here
and in the dashboard.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C


def feature_names() -> list[str]:
    """Fixed, ordered model-input column list."""
    return list(C.NUMERIC_FEATURES) + [f"type_{t}" for t in C.TYPE_CATEGORIES]


N_FEATURES = len(C.NUMERIC_FEATURES) + len(C.TYPE_CATEGORIES)


@dataclass
class Preprocessor:
    """Fitted preprocessing state. Fit once on train, reuse everywhere."""
    scaler: StandardScaler = field(default_factory=StandardScaler)
    numeric_cols: list = field(default_factory=lambda: list(C.NUMERIC_FEATURES))
    type_cats: list = field(default_factory=lambda: list(C.TYPE_CATEGORIES))
    fitted: bool = False

    # ---- core transforms -------------------------------------------------
    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.numeric_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].replace([np.inf, -np.inf], 0.0)
        if "type" not in df.columns:
            df["type"] = self.type_cats[0]
        df["type"] = df["type"].astype(str)
        return df

    def fit(self, df_train: pd.DataFrame) -> "Preprocessor":
        df = self._frame(df_train)
        self.scaler.fit(df[self.numeric_cols].values)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit before transform().")
        df = self._frame(df)
        num = self.scaler.transform(df[self.numeric_cols].values)
        onehot = np.zeros((len(df), len(self.type_cats)), dtype=float)
        cat_index = {c: i for i, c in enumerate(self.type_cats)}
        for i, v in enumerate(df["type"].values):
            if v in cat_index:
                onehot[i, cat_index[v]] = 1.0
            # unknown category -> all-zero row (compatible shape preserved)
        return np.hstack([num, onehot]).astype(float)

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        return self.fit(df_train).transform(df_train)

    # ---- schema helpers ---------------------------------------------------
    def schema(self) -> dict:
        return {
            "numeric": list(self.numeric_cols),
            "type_categories": list(self.type_cats),
            "feature_order": feature_names(),
            "n_features": N_FEATURES,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"scaler": self.scaler, "numeric": self.numeric_cols,
                     "cats": self.type_cats, "fitted": self.fitted}, path)
        with open(path + ".schema.json", "w") as f:
            json.dump(self.schema(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        obj = joblib.load(path)
        p = cls()
        p.scaler = obj["scaler"]
        p.numeric_cols = obj["numeric"]
        p.type_cats = obj["cats"]
        p.fitted = obj["fitted"]
        return p


def validate_feature_matrix(X: np.ndarray, context: str = "") -> None:
    """Fail fast if a bank produced an incompatible matrix."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(f"{context}: expected 2-D numpy array, got {type(X)}")
    if X.shape[1] != N_FEATURES:
        raise ValueError(
            f"{context}: expected {N_FEATURES} features {feature_names()}, "
            f"got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{context}: non-finite values in feature matrix.")
