import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import joblib

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

    # Check if we already have processed data
    metadata_path = os.path.join(data_dir, "metadata.npz")
    if os.path.exists(metadata_path) and not force_download:
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

    print("Generating simulated dataset based on description...")
    df = generate_simulated_dataset(50000)  # Smaller dataset for testing

    print(f"Dataset loaded with {len(df)} transactions")
    print(f"Fraud rate: {df['isFraud'].mean():.4f}")

    # Preprocess the data
    print("Preprocessing data...")
    processed_data = {}

    # Define feature columns (based on PaySim dataset)
    feature_cols = [col for col in df.columns if col not in ['isFraud', 'nameOrig', 'nameDest', 'BankID']]

    # Handle categorical features if any
    # For simplicity, we'll assume most features are already numeric
    # In PaySim, we have: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud, BankID

    # Encode transaction type if it exists
    if 'type' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['type_encoded'] = le.fit_transform(df['type'].astype(str))
        feature_cols = [c if c != 'type' else 'type_encoded' for c in feature_cols]
        if 'type' in feature_cols:
            feature_cols.remove('type')
        feature_cols.append('type_encoded')

    # Select features and target
    X = df[feature_cols].fillna(0)
    y = df['isFraud']

    # Split by BankID
    for bank_id in range(5):
        bank_mask = df['BankID'] == bank_id
        bank_X = X[bank_mask].values
        bank_y = y[bank_mask].values

        # Standardize features (zero mean, unit variance)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        bank_X_scaled = scaler.fit_transform(bank_X)

        processed_data[bank_id] = {
            'X': bank_X_scaled,
            'y': bank_y,
            'scaler': scaler,
            'feature_names': feature_cols,
            'n_samples': len(bank_X)
        }

        print(f"Bank {bank_id}: {len(bank_X)} samples, fraud rate: {bank_y.mean():.4f}")

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

    # Save metadata
    metadata_file = os.path.join(data_dir, "metadata.npz")
    metadata = {
        'n_banks': 5,
        'feature_names': processed_data[0]['feature_names']
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