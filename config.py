# CIPHERMESH — central configuration.
# All experiment defaults live here so every module + the dashboard stay consistent.
from __future__ import annotations

DATASET_ID = "flwrlabs/fed-fraud-paysim-banks"
RANDOM_SEED = 42

# Full HF dataset is ~5.7M rows — too large for a hackathon demo.
# We take a reproducible stratified subset (frauds preserved per bank).
SUBSET_TOTAL = 60_000          # total rows sampled from HF (stratified by BankID+isFraud)
MAX_ROWS_PER_BANK = 20_000     # safety cap per bank after sampling

BANK_IDS = [0, 1, 2, 3, 4]
BANK_NAMES = {0: "Bank A", 1: "Bank B", 2: "Bank C", 3: "Bank D", 4: "Bank E"}

TARGET_COL = "isFraud"
BANK_COL = "BankID"
ID_COLS = ["nameOrig", "nameDest"]          # identifiers — never model inputs

# isFlaggedFraud has only ~14 positives in 5.7M rows; kept as a feature but
# documented as near-constant / potential leakage (see preprocessing docs).
NUMERIC_FEATURES = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFlaggedFraud",
]
CATEGORICAL_FEATURES = ["type"]
TYPE_CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

TEST_SIZE = 0.20               # stratified hold-out per bank

# Model hyper-parameters (same for every bank → compatible shapes)
MODEL_PARAMS = {
    "loss": "log_loss",
    "penalty": "l2",
    "alpha": 0.0001,
    "learning_rate": "optimal",
    "eta0": 0.0,
    "random_state": RANDOM_SEED,
}
LOCAL_MAX_ITER = 50            # SGD passes per federated round (configurable in UI)
N_ROUNDS_DEFAULT = 5
CLIP_NORM_DEFAULT = 1.0
NOISE_MULT_DEFAULT = 0.0
DECISION_THRESHOLD = 0.5

DATA_DIR = "./data"
RESULTS_DIR = "./results"
MODELS_DIR = "./models"
