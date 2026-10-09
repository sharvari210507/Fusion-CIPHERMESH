#!/usr/bin/env python3
"""
Demonstration script showing the complete CIPHERMESH workflow.
"""

import sys
import os
import json
sys.path.append('src')

def demonstrate_ciphermesh():
    print("=" * 60)
    print("🔐 CIPHERMESH DEMONSTRATION")
    print("Privacy-Preserving Fraud Signal Sharing Across Banks")
    print("=" * 60)

    # Import our modules
    from src.federated_learning import FederatedLearningCoordinator

    print("\n1. INITIALIZING SYSTEM")
    print("-" * 30)

    # Create coordinator
    coordinator = FederatedLearningCoordinator(
        data_dir="./demo_data",
        results_dir="./demo_results",
        models_dir="./demo_models"
    )

    print("✅ Federated Learning Coordinator initialized")
    print(f"📊 Loaded data for {len(coordinator.bank_data)} banks")

    # Show data statistics
    total_samples = 0
    total_fraud = 0
    for bank_id in range(5):
        X, y = coordinator.bank_data[bank_id]['X'], coordinator.bank_data[bank_id]['y']
        total_samples += len(X)
        total_fraud += y.sum()
        print(f"   Bank {bank_id}: {len(X):,} transactions, {y.sum():,} fraud ({y.mean():.3%})")

    print(f"\n📈 Total: {total_samples:,} transactions, {total_fraud:,} fraud ({total_fraud/total_samples:.3%})")

    print("\n2. RUNNING FEDERATED LEARNING EXPERIMENT")
    print("-" * 45)

    # Run experiment with privacy controls
    print("🚀 Starting federated learning with:")
    print("   • 3 rounds of training")
    print("   • Gradient clipping norm: 1.0")
    print("   • Noise multiplier: 0.05 (light privacy protection)")
    print("   • All 5 banks participating")

    results = coordinator.run_federated_learning(
        n_rounds=3,
        clip_norm=1.0,
        noise_multiplier=0.05,
        participating_banks=None,  # All banks
        save_results=True
    )

    print("\n3. EXPERIMENT RESULTS")
    print("-" * 20)

    final_metrics = results['final_global_metrics']
    print(f"🎯 Final Model Performance:")
    print(f"   F1 Score:     {final_metrics['f1']:.4f}")
    print(f"   PR-AUC:       {final_metrics['pr_auc']:.4f}")
    print(f"   Precision:    {final_metrics['precision']:.4f}")
    print(f"   Recall:       {final_metrics['recall']:.4f}")

    # Show progression
    print(f"\n📈 Performance Progression:")
    for round_result in results['round_history']:
        round_num = round_result['round']
        f1 = round_result['global_metrics']['f1']
        auc = round_result['global_metrics']['pr_auc']
        print(f"   Round {round_num}: F1={f1:.4f}, PR-AUC={auc:.4f}")

    print("\n4. PRIVACY GUARANTEES")
    print("-" * 22)
    print("🔒 What CIPHERMESH Protects:")
    print("   • Zero raw transaction data shared between banks")
    print("   • Only model weight updates (gradients) exchanged")
    print("   • Sensitive customer information remains local")
    print("   • Configurable privacy controls (clipping + noise)")

    print(f"\n📊 Privacy Metrics for this Experiment:")
    print(f"   • Rows shared: 0 (raw data never leaves banks)")
    print(f"   • Model updates shared: {sum(r['n_samples'] for r in results['round_history'][0]['local_results'])} total samples worth")
    print(f"   • Privacy protection: Gradient clipping (norm=1.0) + Gaussian noise (σ=0.05)")

    print("\n5. FILES GENERATED")
    print("-" * 18)
    print("💾 Results saved to:")
    print(f"   • {os.listdir('./demo_results')}")
    print(f"   • {os.listdir('./demo_models')}")

    # Show a sample of the results
    result_files = [f for f in os.listdir('./demo_results') if f.endswith('.json')]
    if result_files:
        latest_result = max(result_files, key=lambda f: os.path.getctime(os.path.join('./demo_results', f)))
        print(f"\n📄 Sample result file: {latest_result}")
        with open(os.path.join('./demo_results', latest_result), 'r') as f:
            sample_data = json.load(f)
            print(f"   Rounds completed: {len(sample_data['round_history'])}")
            print(f"   Final F1: {sample_data['final_global_metrics']['f1']:.4f}")

    print("\n" + "=" * 60)
    print("🎉 DEMONSTRATION COMPLETE")
    print("CIPHERMESH enables banks to collaboratively improve fraud detection")
    print("while preserving customer privacy through federated learning.")
    print("=" * 60)

    return results

if __name__ == "__main__":
    demonstrate_ciphermesh()