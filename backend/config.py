from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"; MODELS_DIR = ROOT / "models"; RESULTS_DIR = ROOT / "results"
DB_PATH = ROOT / "data" / "fedguard.db"
DATASET_ID = "flwrlabs/fed-fraud-paysim-banks"
LABEL = "isFraud"; BANK_COL = "BankID"; N_BANKS = 5
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
AMOUNT_MAX = 1e9; SEED = 42
AUTO_START_JOBS = True
DEFAULT_RUN = dict(banks=[0, 1, 2, 3, 4], rounds=5, local_epochs=1, feature_mode="raw",
    clip_norm=1.0, noise_multiplier=0.0, sampling_frac=1.0, seed=42)
EXP_DEFAULT = dict(seeds=[42, 43, 44], rounds=5, local_epochs=1)
