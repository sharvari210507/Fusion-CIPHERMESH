# CSB-04 Dataset Audit

## Dataset
Name: fed-fraud-paysim-banks  
Source: https://huggingface.co/datasets/flwrlabs/fed-fraud-paysim-banks

## Published dataset facts
- Training split: 5,726,358 rows
- Test split: 636,262 rows
- Five simulated banks, identified by BankID (0–4)
- Target label: isFraud
- Transaction features include type, amount, step and account balances.
- Account identifiers include nameOrig and nameDest.
- isFlaggedFraud is an existing rule-based flag and requires careful consideration before feature selection.
- The dataset uses synthetic PaySim-style transactions.

## Questions to resolve before implementation
1. What is the fraud prevalence in each bank's training and test partitions?
2. How different are the transaction distributions across banks?
3. Which features will be used for prediction, and why?
4. How will preprocessing remain consistent across all clients?
5. Will the experiment compare local-only models against a federated model on the same held-out test data?
6. How will model updates be protected against potential information leakage?

## Evaluation requirements
Report precision, recall, F1-score and confusion matrices. Consider PR-AUC because fraud detection is a class-imbalanced problem.

## Important limitations
The dataset is synthetic and its bank partitions are simulated. Results demonstrate an experimental prototype, not production banking performance or formal privacy compliance.

## Status
Initial documentation review only. Dataset statistics and experimental results still need to be verified locally.
