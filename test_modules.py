#!/usr/bin/env python3
"""
Test script to verify that all modules work correctly.
"""

import sys
import os


def test_dataset_module():
    """Test the dataset module with simulated data."""
    print("Testing dataset module...")
    from src.dataset import load_and_preprocess_data, get_bank_data

    # Force simulated data generation by using a non-existent dataset name
    # This will trigger the fallback to generate_simulated_dataset
    data = load_and_preprocess_data("./data_test", force_download=True)

    print(f"Loaded data for {len(data)} banks")
    total_samples = 0
    total_fraud = 0

    for bank_id in range(5):
        X, y = get_bank_data(bank_id, data)
        total_samples += len(X)
        total_fraud += y.sum()
        print(f"Bank {bank_id}: {len(X)} samples, {y.sum()} fraud cases ({y.mean():.4f})")

    print(f"Total: {total_samples} samples, {total_fraud} fraud cases ({total_fraud/total_samples:.4f})")
    return data

def test_local_training_module(data):
    """Test the local training module."""
    print("\nTesting local training module...")
    from src.local_training import train_local_model
    from src.dataset import get_bank_data

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

    print(f"Training completed:")
    print(f"  Train metrics: {result['train_metrics']}")
    print(f"  Val metrics: {result['val_metrics']}")
    print(f"  Weight norm: {np.linalg.norm(result['weights']['coef']):.4f}")
    return result

def test_federated_learning_module():
    """Test the federated learning coordinator."""
    print("\nTesting federated learning module...")
    from src.federated_learning import FederatedLearningCoordinator

    # Create coordinator with test directories
    coordinator = FederatedLearningCoordinator(
        data_dir="./data_test",
        results_dir="./results_test",
        models_dir="./models_test"
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

    print(f"Federated learning completed!")
    print(f"Final global metrics: {results['final_global_metrics']}")
    return results

if __name__ == "__main__":
    import numpy as np

    print("=" * 50)
    print("CIPHERMESH Module Testing")
    print("=" * 50)

    try:
        # Test dataset module
        data = test_dataset_module()

        # Test local training module
        local_result = test_local_training_module(data)

        # Test federated learning module
        fl_results = test_federated_learning_module()

        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)