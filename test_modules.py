#!/usr/bin/env python3
"""
Test script to verify that all modules work correctly.
"""

import sys
import os
import numpy as np


def test_dataset_module(tmp_path):
    """Test the dataset module with simulated data."""
    print("Testing dataset module...")
    from src.dataset import load_and_preprocess_data, get_bank_data
    from src.preprocessing import feature_names as expected_names

    data = load_and_preprocess_data(str(tmp_path / "data"), force_download=True)

    assert set(data.keys()) == {0, 1, 2, 3, 4}, "expected five bank partitions"
    total_samples = 0
    total_fraud = 0
    for bank_id in range(5):
        X, y = get_bank_data(bank_id, data)
        assert X.ndim == 2 and X.shape[1] == len(expected_names())
        assert data[bank_id]["feature_names"] == expected_names()
        assert set(np.unique(y)) <= {0, 1}
        total_samples += len(X)
        total_fraud += y.sum()
        print(f"Bank {bank_id}: {len(X)} samples, {y.sum()} fraud cases ({y.mean():.4f})")

    assert total_samples > 0
    assert total_fraud > 0, "simulated data must contain fraud cases"
    print(f"Total: {total_samples} samples, {total_fraud} fraud cases ({total_fraud/total_samples:.4f})")

def test_local_training_module(tmp_path):
    """Test the local training module."""
    print("\nTesting local training module...")
    from src.local_training import train_local_model
    from src.dataset import load_and_preprocess_data, get_bank_data

    data = load_and_preprocess_data(str(tmp_path / "data"), force_download=True)

    # Get data for bank 0
    X, y = get_bank_data(0, data)

    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train model
    result = train_local_model(
        X_train, y_train,
        X_val, y_val,
        clip_norm=1.0,
        noise_multiplier=0.1,
        random_state=42
    )

    assert "train_metrics" in result and "val_metrics" in result
    assert "weights" in result and np.all(np.isfinite(result["weights"]["coef"]))
    for split in ("train_metrics", "val_metrics"):
        for k in ("pr_auc", "f1"):
            assert 0.0 <= result[split][k] <= 1.0, f"{split}[{k}] out of range"
    print(f"Training completed:")
    print(f"  Train metrics: {result['train_metrics']}")
    print(f"  Val metrics: {result['val_metrics']}")
    print(f"  Weight norm: {np.linalg.norm(result['weights']['coef']):.4f}")

def test_federated_learning_module(tmp_path):
    """Test the federated learning coordinator."""
    print("\nTesting federated learning module...")
    from src.federated_learning import FederatedLearningCoordinator

    # Create coordinator with isolated scratch directories
    coordinator = FederatedLearningCoordinator(
        data_dir=str(tmp_path / "data"),
        results_dir=str(tmp_path / "results"),
        models_dir=str(tmp_path / "models")
    )

    print("Coordinator created successfully!")

    # Test with just 2 rounds for quick testing
    results = coordinator.run_federated_learning(
        n_rounds=2,
        clip_norm=1.0,
        noise_multiplier=0.0,
        participating_banks=[0, 1],  # Just two banks for testing
        save_results=True
    )

    assert results["n_rounds"] == 2
    assert len(results["round_history"]) == 2
    assert "final_global_metrics" in results
    assert 0.0 <= results["final_global_metrics"]["pr_auc"] <= 1.0
    print(f"Federated learning completed!")
    print(f"Final global metrics: {results['final_global_metrics']}")

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    print("=" * 50)
    print("CIPHERMESH Module Testing")
    print("=" * 50)

    scratch = Path(tempfile.mkdtemp(prefix="ciphermesh_mods_"))
    try:
        # Test dataset module
        test_dataset_module(scratch / "data")

        # Test local training module
        test_local_training_module(scratch / "data")

        # Test federated learning module
        test_federated_learning_module(scratch)

        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)