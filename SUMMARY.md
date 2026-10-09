# CIPHERMESH - Privacy-Preserving Fraud Signal Sharing Across Banks

## Overview
CIPHERMESH is a federated learning solution that enables multiple banks to collaboratively train a fraud detection model without sharing raw transaction data. The system uses Federated Averaging (FedAvg) to combine model updates from each bank while preserving data privacy.

## Key Features
- **Privacy-Preserving**: No raw transaction data leaves participating banks
- **Federated Learning**: Implements Federated Averaging (FedAvg) for collaborative model training
- **Privacy Controls**: Configurable gradient clipping and Gaussian noise for differential privacy
- **Interactive Dashboard**: Streamlit-based web interface for running experiments and visualizing results
- **Experiment Tracking**: Automatic saving of results, models, and comparison data
- **Performance Evaluation**: Measures F1-score, PR-AUC, precision, and recall on held-out test data

## System Components
1. **Dataset Module** (`src/dataset.py`): 
   - Loads and preprocesses federated fraud detection data
   - Simulates PaySim-like dataset when Hugging Face dataset unavailable
   - Splits data by BankID and applies standardization

2. **Local Training Module** (`src/local_training.py`):
   - Trains logistic regression models using SGDClassifier
   - Implements gradient clipping for update normalization
   - Adds Gaussian noise for differential privacy
   - Handles class imbalance for fraud detection

3. **Federated Learning Coordinator** (`src/federated_learning.py`):
   - Manages federated learning process across multiple banks
   - Implements Federated Averaging (FedAvg) for model aggregation
   - Provides privacy controls (clipping + noise)
   - Evaluates global model on held-out test set
   - Saves results and models automatically

4. **Web Dashboard** (`dashboard.py`):
   - Streamlit-based interactive interface
   - Federation Control Room: Run and monitor experiments
   - Experiment Results: Compare different configurations
   - Privacy Audit: Visualize privacy guarantees and metrics
   - Try a Transaction: Demo fraud detection on sample transactions

## Installation & Usage

### Prerequisites
- Python 3.14+
- pip (Python package installer)

### Setup
```bash
# Clone repository (if not already done)
git clone <repository-url>
cd Fusion-CIPHERMESH

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

#### Command Line Experiments
```bash
# Quick test
source venv/bin/activate
python3 -c "
import sys
sys.path.append('src')
from federated_learning import FederatedLearningCoordinator
coordinator = FederatedLearningCoordinator('./data_test', './results_test', './models_test')
results = coordinator.run_federated_learning(n_rounds=3, participating_banks=[0,1])
print('Final F1:', results['final_global_metrics']['f1'])
"

# Full experiment comparison
source venv/bin/activate
python3 -c "
import sys
sys.path.append('src')
from federated_learning import run_standalone_experiment
run_standalone_experiment()
"
```

#### Web Dashboard
```bash
source venv/bin/activate
python3 run_dashboard.py
```
Then open your browser to: http://localhost:8501

## Project Structure
```
Fusion-CIPHERMESH/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Data loading and preprocessing
│   ├── local_training.py   # Local model training with privacy controls
│   └── federated_learning.py # Federated learning coordinator
├── dashboard.py            # Streamlit web interface
├── run_dashboard.py        # Dashboard launcher script
├── demo.py                 # Complete system demonstration
├── requirements.txt        # Python dependencies
├── README.md               # Detailed documentation
└── SUMMARY.md              # This summary file
```

## Privacy Guarantees
- **Zero Raw Data Sharing**: Only model weight updates (gradients) are exchanged between banks
- **Local Data Processing**: All transaction data remains on each bank's premises
- **Configurable Privacy**: Adjustable gradient clipping norm and noise multiplier
- **Transparent Audit**: Clear visibility into what information is shared (see Privacy Audit dashboard)

## Performance Metrics
The system evaluates model performance using:
- **F1 Score**: Harmonic mean of precision and recall (better for imbalanced data)
- **PR-AUC**: Area under Precision-Recall curve (appropriate for ~0.1% fraud rate)
- **Precision**: Percentage of flagged transactions that are actually fraudulent
- **Recall**: Percentage of fraudulent transactions correctly identified

## Files Generated
- **Results**: JSON files in `./results/` containing round-by-round metrics
- **Models**: NumPy archives in `./models/` containing global model weights
- **Experiment Comparisons**: JSON files comparing different privacy-utility trade-offs

## Future Enhancements
Planned improvements for production deployment:
1. **Secure Aggregation**: Cryptographic protocols to protect update privacy
2. **TLS Encryption**: Secure communication channels between banks and coordinator
3. **Access Control**: Authentication and authorization for participating institutions
4. **Formal Privacy Accounting**: Rigorous differential privacy budget tracking
5. **Real Dataset Integration**: Connection to actual banking transaction data (with proper permissions)
6. **Model Interpretability**: Feature importance explanations for fraud predictions

## Acknowledgements
This system was created for the CSB-04 challenge: Privacy-Preserving Fraud Signal Sharing Across Banks.

**Together • Smarter detection • Data stays home.**