import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import joblib

# Cache schema version. Bump whenever the cached feature layout changes; caches
# written by older code are rebuilt instead of silently reused.
CACHE_SCHEMA_VERSION = 2


def expected_feature_names():
    """The single source of truth lives in src/preprocessing.feature_names():
    7 numeric columns + one-hot `type` in fixed order (12 total)."""
    from src.preprocessing import feature_names
    return feature_names()


def _cache_valid(data_dir):
    """A cache is usable only if its schema version and feature list exactly
    match the current pipeline. Anything else -> rebuild, never silent reuse."""
    try:
        meta_path = os.path.join(data_dir, "metadata.npz")
        if not os.path.exists(meta_path):
            return False
        metadata = np.load(meta_path, allow_pickle=True)
        if "schema_version" not in metadata.files:
            return False
        if int(metadata["schema_version"]) != CACHE_SCHEMA_VERSION:
            return False
        if list(metadata["feature_names"]) != expected_feature_names():
            return False
        n_banks = int(metadata["n_banks"])
        for bank_id in range(n_banks):
            bank_file = os.path.join(data_dir, f"bank_{bank_id}_data.npz")
            if not os.path.exists(bank_file):
                return False
            b = np.load(bank_file, allow_pickle=True)
            if b["X"].ndim != 2 or b["X"].shape[1] != len(expected_feature_names()):
                return False
            if list(b["feature_names"]) != expected_feature_names():
                return False
            if not os.path.exists(os.path.join(data_dir, f"bank_{bank_id}_scaler.save")):
                return False
        return True
    except Exception:
        return False

def load_and_preprocess_data(data_dir="./data", force_download=False):
    """
    Load the federated fraud detection dataset and preprocess it for each bank.

    Args:
        data_dir: Directory to store/load data
        force_download: Whether to force re-download even if cached

    Returns:
        dict: Dictionary containing data for each bank (0-4)
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if we already have processed data with a compatible schema
    if not force_download and _cache_valid(data_dir):
        metadata_path = os.path.join(data_dir, "metadata.npz")
        print("Loading preprocessed data from cache...")
        # Load metadata
        metadata = np.load(metadata_path)
        n_banks = int(metadata['n_banks'])
        feature_names = metadata['feature_names'].tolist()

        # Load each bank's data
        processed_data = {}
        for bank_id in range(n_banks):
            # Load bank data
            bank_file = os.path.join(data_dir, f"bank_{bank_id}_data.npz")
            bank_data = np.load(bank_file)

            # Load scaler
            scaler_file = os.path.join(data_dir, f"bank_{bank_id}_scaler.save")
            scaler = joblib.load(scaler_file)

            processed_data[bank_id] = {
                'X': bank_data['X'],
                'y': bank_data['y'],
                'scaler': scaler,
                'feature_names': bank_data['feature_names'].tolist(),
                'n_samples': len(bank_data['X'])
            }

            print(f"Bank {bank_id}: {len(bank_data['X'])} samples, fraud rate: {bank_data['y'].mean():.4f}")

        return processed_data

    if os.path.exists(os.path.join(data_dir, "metadata.npz")) and not _cache_valid(data_dir):
        print("Cached data schema mismatch (stale version or unexpected feature "
              "columns). Rebuilding cache from the current pipeline schema; "
              "stale files are overwritten, never silently reused.")

    print("Generating simulated dataset based on description...")
    df = generate_simulated_dataset(50000)  # Smaller dataset for testing

    print(f"Dataset loaded with {len(df)} transactions")
    print(f"Fraud rate: {df['isFraud'].mean():.4f}")

    # Preprocess the data
    print("Preprocessing data...")
    processed_data = {}

    # Feature layout follows the pipeline schema exactly: numeric columns plus
    # one-hot `type` in the fixed category order (no duplicate columns).
    # Column identity comes from the preprocessing schema, not local literals.
    from src.preprocessing import Preprocessor as _Pre
    _ref = _Pre()
    numeric_cols = list(_ref.numeric_cols)
    type_cats = list(_ref.type_cats)
    feature_cols = expected_feature_names()

    # Select features and target
    y = df['isFraud']

    # Split by BankID
    for bank_id in range(5):
        bank_mask = df['BankID'] == bank_id
        bank_df = df[bank_mask]
        bank_y = y[bank_mask].values

        # Standardize numeric features (zero mean, unit variance)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(bank_df[numeric_cols].fillna(0).values.astype(float))
        cats = bank_df["type"].astype(str).values
        onehot = np.zeros((len(bank_df), len(type_cats)), dtype=float)
        idx = {c: i for i, c in enumerate(type_cats)}
        for i, v in enumerate(cats):
            if v in idx:
                onehot[i, idx[v]] = 1.0
        bank_X_scaled = np.hstack([num_scaled, onehot])

        processed_data[bank_id] = {
            'X': bank_X_scaled,
            'y': bank_y,
            'scaler': scaler,
            'feature_names': feature_cols,
            'n_samples': len(bank_X_scaled)
        }

        print(f"Bank {bank_id}: {len(bank_X_scaled)} samples, fraud rate: {bank_y.mean():.4f}")

    # Cache the processed data
    print("Caching processed data...")
    # Save each bank's data separately since np.savez doesn't handle nested dicts well
    for bank_id in range(5):
        bank_file = os.path.join(data_dir, f"bank_{bank_id}_data.npz")
        bank_data = processed_data[bank_id]
        # Save arrays and convert scaler to parameters
        np.savez(bank_file,
                 X=bank_data['X'],
                 y=bank_data['y'],
                 feature_names=bank_data['feature_names'])
        # Save scaler separately
        scaler_file = os.path.join(data_dir, f"bank_{bank_id}_scaler.save")
        joblib.dump(bank_data['scaler'], scaler_file)

    # Save metadata (schema version + exact feature list for validation)
    metadata_file = os.path.join(data_dir, "metadata.npz")
    metadata = {
        'n_banks': 5,
        'feature_names': processed_data[0]['feature_names'],
        'schema_version': CACHE_SCHEMA_VERSION,
    }
    np.savez(metadata_file, **metadata)

    return processed_data

def generate_simulated_dataset(n_samples=10000):
    """
    Generate a simulated dataset similar to PaySim for testing purposes.
    """
    np.random.seed(42)

    # Generate basic transaction features
    data = {
        'step': np.random.randint(1, 744, n_samples),  # hours in a month
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT'], n_samples),
        'amount': np.random.exponential(1000, n_samples),
        'oldbalanceOrg': np.random.exponential(5000, n_samples),
        'newbalanceOrig': np.random.exponential(5000, n_samples),
        'oldbalanceDest': np.random.exponential(5000, n_samples),
        'newbalanceDest': np.random.exponential(5000, n_samples),
        'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    }

    df = pd.DataFrame(data)

    # Assign BankID (non-IID distribution)
    df['BankID'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2])

    # Generate fraud labels with some correlation to transaction types and amounts
    # Make certain transaction types more likely to be fraudulent
    fraud_prob = np.zeros(n_samples)

    # Increase fraud probability for certain types
    fraud_types = ['TRANSFER', 'CASH_OUT']
    for t in fraud_types:
        fraud_prob[df['type'] == t] += 0.05

    # Increase fraud probability for high amounts
    fraud_prob += (df['amount'] > df['amount'].quantile(0.95)) * 0.1

    # Increase fraud probability for balance inconsistencies
    # For outgoing transactions: newbalanceOrig should be <= oldbalanceOrg
    # For incoming transactions: newbalanceDest should be >= oldbalanceDest
    # We'll simulate some inconsistency as potential fraud
    outflow_tx = df['type'].isin(['TRANSFER', 'CASH_OUT', 'PAYMENT'])
    inflow_tx = df['type'] == 'DEBIT'

    # Outflow inconsistency: money appeared in origin account
    outflow_issue = outflow_tx & (df['newbalanceOrig'] > df['oldbalanceOrg'] + df['amount'] * 0.1)
    # Inflow inconsistency: money disappeared from destination account
    inflow_issue = inflow_tx & (df['newbalanceDest'] < df['oldbalanceDest'] - df['amount'] * 0.1)

    balance_issue = outflow_issue | inflow_issue
    fraud_prob[balance_issue] += 0.1

    # Normalize and ensure probabilities are in [0, 1]
    fraud_prob = np.clip(fraud_prob, 0, 0.3)  # Cap at 30% fraud probability

    # Generate fraud labels
    df['isFraud'] = np.random.binomial(1, fraud_prob)

    # Adjust to get ~0.1% fraud rate as mentioned in description
    target_fraud_rate = 0.001
    current_fraud_rate = df['isFraud'].mean()
    if current_fraud_rate > 0:
        # Scale down fraud cases to match target rate
        fraud_indices = df[df['isFraud'] == 1].index
        n_to_keep = int(len(fraud_indices) * target_fraud_rate / current_fraud_rate)
        if n_to_keep < len(fraud_indices):
            to_remove = np.random.choice(fraud_indices, len(fraud_indices) - n_to_keep, replace=False)
            df.loc[to_remove, 'isFraud'] = 0

    return df

def get_bank_data(bank_id, processed_data=None):
    """
    Get data for a specific bank.

    Args:
        bank_id: Bank identifier (0-4)
        processed_data: Preprocessed data dictionary (if None, will load)

    Returns:
        tuple: (X, y) features and labels for the bank
    """
    if processed_data is None:
        processed_data = load_and_preprocess_data()

    bank_data = processed_data[bank_id]
    return bank_data['X'], bank_data['y']

if __name__ == "__main__":
    # Test the dataset loading
    print("Testing dataset loading...")
    data = load_and_preprocess_data("./data", force_download=True)

    print("\nDataset Summary:")
    total_samples = 0
    total_fraud = 0
    for bank_id in range(5):
        X, y = get_bank_data(bank_id, data)
        total_samples += len(X)
        total_fraud += np.sum(y)
        print(f"Bank {bank_id}: {len(X)} samples, {np.sum(y)} fraud cases ({np.mean(y):.4f})")

    print(f"\nTotal: {total_samples} samples, {total_fraud} fraud cases ({total_fraud/total_samples:.4f})")