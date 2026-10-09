import numpy as np
import json
import os
from typing import List, Dict, Tuple
from .local_training import train_local_model, aggregate_weights, set_model_weights, evaluate_model
from .dataset import load_and_preprocess_data, get_bank_data

class FederatedLearningCoordinator:
    """
    Coordinator for federated learning across multiple banks.
    Implements Federated Averaging (FedAvg) algorithm.
    """

    def __init__(self, n_banks=5, data_dir="./data", results_dir="./results", models_dir="./models"):
        self.n_banks = n_banks
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.models_dir = models_dir

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Load and preprocess data
        print("Loading and preprocessing data...")
        self.bank_data = load_and_preprocess_data(data_dir)

        # Prepare test set (held-out data from all banks)
        self.prepare_test_set()

        # Initialize global model
        self.global_model = None
        self.round_results = []

    def prepare_test_set(self):
        """Prepare a held-out test set from all banks' data."""
        X_list = []
        y_list = []

        for bank_id in range(self.n_banks):
            X, y = get_bank_data(bank_id, self.bank_data)
            # Use 20% of each bank's data for testing
            split_idx = int(0.8 * len(X))
            X_list.append(X[split_idx:])
            y_list.append(y[split_idx:])

        self.X_test = np.vstack(X_list)
        self.y_test = np.hstack(y_list)
        print(f"Test set prepared: {len(self.X_test)} samples")

    def train_local_round(self, bank_ids=None, clip_norm=None, noise_multiplier=0.0, random_state=42):
        """
        Train one round of local models for selected banks.

        Args:
            bank_ids: List of bank IDs to participate (None for all)
            clip_norm: Gradient clipping norm
            noise_multiplier: Noise multiplier for DP
            random_state: Random seed

        Returns:
            List of training results from each bank
        """
        if bank_ids is None:
            bank_ids = list(range(self.n_banks))

        results = []

        for bank_id in bank_ids:
            print(f"Training Bank {bank_id}...")
            X, y = get_bank_data(bank_id, self.bank_data)

            # Split bank data into train/validation (80/20)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train local model
            result = train_local_model(
                X_train, y_train,
                X_val, y_val,
                clip_norm=clip_norm,
                noise_multiplier=noise_multiplier,
                random_state=random_state + bank_id  # Different seed for each bank
            )

            result['bank_id'] = bank_id
            results.append(result)

            print(f"  Bank {bank_id}: Train F1 = {result['train_metrics']['f1']:.4f}, "
                  f"Val F1 = {result['val_metrics']['f1'] if result['val_metrics'] else 0:.4f}")

        return results

    def aggregate_and_update_global(self, local_results):
        """
        Aggregate local model updates and update global model.

        Args:
            local_results: List of results from local training

        Returns:
            Aggregated weights
        """
        # Extract weights and sample counts
        weight_list = [result['weights'] for result in local_results]
        sample_counts = [result['n_samples'] for result in local_results]

        # Aggregate weights using FedAvg
        aggregated_weights = aggregate_weights(weight_list, sample_counts)

        # Create or update global model
        if self.global_model is None:
            # Create a model with the right shape
            from sklearn.linear_model import SGDClassifier
            import numpy as np
            # Initialize with dummy data to set up internal attributes correctly
            dummy_coef = np.zeros_like(weight_list[0]['coef'])
            dummy_intercept = np.zeros_like(weight_list[0]['intercept'])
            # For binary classification, coef_ shape is (1, n_features)
            n_features = dummy_coef.shape[1]
            dummy_X = np.zeros((2, n_features))  # 2 samples, n_features
            dummy_y = np.array([0, 1])  # Binary classes
            self.global_model = SGDClassifier(loss='log_loss', random_state=42)
            self.global_model.fit(dummy_X, dummy_y)
            # Now set the weights to our initial values (zeros)
            self.global_model.coef_ = dummy_coef
            self.global_model.intercept_ = dummy_intercept

        # Set the aggregated weights
        self.global_model = set_model_weights(self.global_model, aggregated_weights)

        return aggregated_weights

    def evaluate_global_model(self):
        """
        Evaluate the global model on the held-out test set.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.global_model is None:
            return {'error': 'No global model available'}

        metrics = evaluate_model(self.global_model, self.X_test, self.y_test)
        return metrics

    def run_federated_learning(self, n_rounds=10, clip_norm=None, noise_multiplier=0.0,
                             participating_banks=None, save_results=True):
        """
        Run federated learning for multiple rounds.

        Args:
            n_rounds: Number of federated learning rounds
            clip_norm: Gradient clipping norm for DP
            noise_multiplier: Noise multiplier for DP
            participating_banks: List of bank IDs to participate each round (None for all)
            save_results: Whether to save results to disk

        Returns:
            Dictionary containing rounds results and final metrics
        """
        print(f"Starting Federated Learning for {n_rounds} rounds...")
        print(f"Parameters: clip_norm={clip_norm}, noise_multiplier={noise_multiplier}")

        round_history = []

        for round_num in range(n_rounds):
            print(f"\n--- Round {round_num + 1}/{n_rounds} ---")

            # Train local models
            local_results = self.train_local_round(
                bank_ids=participating_banks,
                clip_norm=clip_norm,
                noise_multiplier=noise_multiplier,
                random_state=42 + round_num
            )

            # Aggregate and update global model
            aggregated_weights = self.aggregate_and_update_global(local_results)

            # Evaluate global model
            global_metrics = self.evaluate_global_model()

            # Record round results
            round_result = {
                'round': round_num + 1,
                'local_results': [
                    {
                        'bank_id': r['bank_id'],
                        'train_metrics': r['train_metrics'],
                        'val_metrics': r['val_metrics'],
                        'n_samples': r['n_samples']
                    }
                    for r in local_results
                ],
                'global_metrics': global_metrics,
                'aggregated_weight_norm': {
                    'coef': np.linalg.norm(aggregated_weights['coef']),
                    'intercept': np.linalg.norm(aggregated_weights['intercept'])
                }
            }

            round_history.append(round_result)

            print(f"Global Model - Test F1: {global_metrics['f1']:.4f}, "
                  f"Test PR-AUC: {global_metrics['pr_auc']:.4f}")

        # Final results
        final_results = {
            'n_rounds': n_rounds,
            'clip_norm': clip_norm,
            'noise_multiplier': noise_multiplier,
            'participating_banks': participating_banks if participating_banks is not None else list(range(self.n_banks)),
            'round_history': round_history,
            'final_global_metrics': self.evaluate_global_model()
        }

        # Save results if requested
        if save_results:
            self.save_results(final_results)
            self.save_global_model()

        return final_results

    def save_results(self, results):
        """Save results to JSON file."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"fl_results_{timestamp}.json")

        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        json_results = convert_for_json(results)

        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {filename}")

    def save_global_model(self):
        """Save the global model weights."""
        if self.global_model is None:
            print("No global model to save")
            return

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.models_dir, f"global_model_{timestamp}.npz")

        # Save model weights
        np.savez(
            filename,
            coef=self.global_model.coef_,
            intercept=self.global_model.intercept_
        )

        print(f"Global model saved to {filename}")

    def load_global_model(self, model_path):
        """Load a saved global model."""
        data = np.load(model_path)
        from sklearn.linear_model import SGDClassifier
        self.global_model = SGDClassifier(loss='log_loss', random_state=42)
        self.global_model.coef_ = data['coef']
        self.global_model.intercept_ = data['intercept']
        print(f"Global model loaded from {model_path}")

def run_standalone_experiment():
    """Run a standalone federated learning experiment for demonstration."""
    print("=== CIPHERMESH Federated Learning Experiment ===")

    # Initialize coordinator
    coordinator = FederatedLearningCoordinator()

    # Experiment configurations
    experiments = [
        {
            'name': 'Local Only (Baseline)',
            'n_rounds': 1,
            'clip_norm': None,
            'noise_multiplier': 0.0,
            'participating_banks': [0]  # Only bank 0
        },
        {
            'name': 'Federated (2 Banks)',
            'n_rounds': 5,
            'clip_norm': None,
            'noise_multiplier': 0.0,
            'participating_banks': [0, 1]
        },
        {
            'name': 'Federated (All Banks)',
            'n_rounds': 10,
            'clip_norm': None,
            'noise_multiplier': 0.0,
            'participating_banks': None  # All banks
        },
        {
            'name': 'Federated + Privacy (Clipping)',
            'n_rounds': 10,
            'clip_norm': 1.0,
            'noise_multiplier': 0.0,
            'participating_banks': None
        },
        {
            'name': 'Federated + Privacy (Clipping + Noise)',
            'n_rounds': 10,
            'clip_norm': 1.0,
            'noise_multiplier': 0.1,
            'participating_banks': None
        }
    ]

    all_results = {}

    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Running Experiment: {exp['name']}")
        print(f"{'='*50}")

        results = coordinator.run_federated_learning(
            n_rounds=exp['n_rounds'],
            clip_norm=exp['clip_norm'],
            noise_multiplier=exp['noise_multiplier'],
            participating_banks=exp['participating_banks'],
            save_results=True
        )

        all_results[exp['name']] = results

        # Print summary
        final_metrics = results['final_global_metrics']
        print(f"\n{exp['name']} Summary:")
        print(f"  Final Test F1: {final_metrics['f1']:.4f}")
        print(f"  Final Test PR-AUC: {final_metrics['pr_auc']:.4f}")
        print(f"  Final Test Precision: {final_metrics['precision']:.4f}")
        print(f"  Final Test Recall: {final_metrics['recall']:.4f}")

    # Save comparison results
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(coordinator.results_dir, f"experiment_comparison_{timestamp}.json")

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    json_results = convert_for_json(all_results)
    with open(comparison_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nExperiment comparison saved to {comparison_file}")

    return all_results

if __name__ == "__main__":
    # Run standalone experiment
    run_standalone_experiment()