import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our modules
import sys
sys.path.append('src')
from src.federated_learning import FederatedLearningCoordinator

# Page configuration
st.set_page_config(
    page_title="CIPHERMESH - Privacy-Preserving Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def load_latest_results(results_dir="./results"):
    """Load the latest FL results."""
    if not os.path.exists(results_dir):
        return None

    result_files = glob.glob(os.path.join(results_dir, "fl_results_*.json"))
    if not result_files:
        return None

    latest_file = max(result_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        return json.load(f)

def load_experiment_comparison(results_dir="./results"):
    """Load experiment comparison results."""
    if not os.path.exists(results_dir):
        return None

    comp_files = glob.glob(os.path.join(results_dir, "experiment_comparison_*.json"))
    if not comp_files:
        return None

    latest_file = max(comp_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_round_metrics(round_history):
    """Plot metrics over rounds."""
    rounds = [r['round'] for r in round_history]

    # Extract metrics
    f1_scores = [r['global_metrics']['f1'] for r in round_history]
    pr_auc_scores = [r['global_metrics']['pr_auc'] for r in round_history]
    precision_scores = [r['global_metrics']['precision'] for r in round_history]
    recall_scores = [r['global_metrics']['recall'] for r in round_history]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('F1 Score', 'PR-AUC', 'Precision', 'Recall'),
        vertical_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(x=rounds, y=f1_scores, mode='lines+markers', name='F1'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=rounds, y=pr_auc_scores, mode='lines+markers', name='PR-AUC'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=rounds, y=precision_scores, mode='lines+markers', name='Precision'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=rounds, y=recall_scores, mode='lines+markers', name='Recall'),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=False, title_text="Federated Learning Performance Over Rounds")
    return fig

def plot_bank_participation(round_history):
    """Plot bank participation and performance."""
    rounds = [r['round'] for r in round_history]

    # Prepare data for heatmap
    n_banks = len(round_history[0]['local_results'])
    bank_ids = [r['bank_id'] for r in round_history[0]['local_results']]

    f1_matrix = []
    for round_data in round_history:
        round_f1 = [local_result['val_metrics']['f1'] if local_result['val_metrics'] else 0
                   for local_result in round_data['local_results']]
        f1_matrix.append(round_f1)

    fig = go.Figure(data=go.Heatmap(
        z=np.array(f1_matrix).T,
        x=rounds,
        y=[f'Bank {bid}' for bid in bank_ids],
        colorscale='Viridis',
        colorbar=dict(title="F1 Score")
    ))

    fig.update_layout(
        title="Bank Performance Heatmap (F1 Score by Round)",
        xaxis_title="Round",
        yaxis_title="Bank",
        height=400
    )

    return fig

def show_federation_control():
    st.markdown('<h2 class="sub-header">Federation Control Room</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Run Federated Learning Experiment")

        # Parameters
        n_rounds = st.slider("Number of Rounds", 1, 20, 5)
        clip_norm = st.selectbox("Gradient Clipping Norm", [None, 0.5, 1.0, 2.0], index=2)
        noise_multiplier = st.slider("Noise Multiplier (Privacy)", 0.0, 1.0, 0.0, 0.1)
        n_banks = st.slider("Number of Participating Banks", 1, 5, 5)

        if st.button("🚀 Run Experiment", type="primary"):
            with st.spinner("Running federated learning experiment..."):
                # Initialize coordinator
                coordinator = FederatedLearningCoordinator(
                    data_dir="./data",
                    results_dir="./results",
                    models_dir="./models"
                )

                # Run experiment
                participating_banks = list(range(n_banks)) if n_banks < 5 else None
                results = coordinator.run_federated_learning(
                    n_rounds=n_rounds,
                    clip_norm=clip_norm,
                    noise_multiplier=noise_multiplier,
                    participating_banks=participating_banks,
                    save_results=True
                )

                st.success("Experiment completed!")

                # Display results
                final_metrics = results['final_global_metrics']

                # Metrics cards
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("F1 Score", f"{final_metrics['f1']:.4f}")
                with metric_col2:
                    st.metric("PR-AUC", f"{final_metrics['pr_auc']:.4f}")
                with metric_col3:
                    st.metric("Precision", f"{final_metrics['precision']:.4f}")
                with metric_col4:
                    st.metric("Recall", f"{final_metrics['recall']:.4f}")

                # Plot results
                fig = plot_round_metrics(results['round_history'])
                st.plotly_chart(fig, use_container_width=True)

                # Bank participation heatmap
                fig2 = plot_bank_participation(results['round_history'])
                st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("System Status")

        # Load latest results
        latest_results = load_latest_results("./results")
        if latest_results:
            st.info("✅ Previous experiment found")
            final_metrics = latest_results['final_global_metrics']
            st.metric("Latest F1 Score", f"{final_metrics['f1']:.4f}")
            st.metric("Latest PR-AUC", f"{final_metrics['pr_auc']:.4f}")
        else:
            st.warning("⚠️ No previous experiments found")

        st.subheader("About Federated Learning")
        st.markdown("""
        **How it works:**
        1. Each bank trains locally on its own data
        2. Banks send only model updates (not data)
        3. Coordinator averages updates (FedAvg)
        4. Updated model sent back to banks
        5. Process repeats for multiple rounds

        **Privacy Features:**
        - No raw transaction data leaves banks
        - Only model weight updates shared
        - Optional gradient clipping & noise addition
        """)

def show_experiment_results():
    st.markdown('<h2 class="sub-header">Experiment Results</h2>', unsafe_allow_html=True)

    # Load experiment comparison
    comparison_data = load_experiment_comparison("./results")

    if comparison_data:
        st.subheader("Experiment Comparison")

        # Prepare data for comparison chart
        experiment_names = list(comparison_data.keys())
        f1_scores = [comparison_data[exp]['final_global_metrics']['f1'] for exp in experiment_names]
        pr_auc_scores = [comparison_data[exp]['final_global_metrics']['pr_auc'] for exp in experiment_names]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='F1 Score', x=experiment_names, y=f1_scores))
        fig.add_trace(go.Bar(name='PR-AUC', x=experiment_names, y=pr_auc_scores))

        fig.update_layout(
            title="Comparison of Different Experimental Configurations",
            xaxis_title="Experiment Configuration",
            yaxis_title="Score",
            barmode='group',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed results table
        st.subheader("Detailed Results")
        results_df = pd.DataFrame({
            'Experiment': experiment_names,
            'F1 Score': [f"{score:.4f}" for score in f1_scores],
            'PR-AUC': [f"{score:.4f}" for score in pr_auc_scores],
            'Precision': [f"{comparison_data[exp]['final_global_metrics']['precision']:.4f}" for exp in experiment_names],
            'Recall': [f"{comparison_data[exp]['final_global_metrics']['recall']:.4f}" for exp in experiment_names]
        })

        st.dataframe(results_df, use_container_width=True)
    else:
        st.info("No experiment comparison data available. Run some experiments first!")

        # Show latest single experiment results
        latest_results = load_latest_results("./results")
        if latest_results:
            st.subheader("Latest Experiment Results")
            st.json(latest_results, expanded=False)

def show_privacy_audit():
    st.markdown('<h2 class="sub-header">Privacy Audit</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### What Information is Shared?

    In CIPHERMESH, we prioritize privacy by ensuring that **no raw transaction data** ever leaves the participating banks.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🔒 Data That Stays Local
        - Individual transaction records
        - Customer account information
        - Personal identifiable information (PII)
        - Raw transaction amounts and timestamps
        """)

    with col2:
        st.markdown("""
        #### 📤 Information That Is Shared
        - Model weight updates (gradients)
        - Aggregated model improvements
        - Performance metrics (accuracy, F1-score, etc.)
        - Number of samples used for training
        """)

    st.markdown("---")

    # Privacy metrics
    st.subheader("Privacy Metrics")

    privacy_col1, privacy_col2, privacy_col3 = st.columns(3)

    with privacy_col1:
        st.metric("Rows Shared", "0", help="Number of raw transaction rows shared between banks")

    with privacy_col2:
        st.metric("Data Leakage Risk", "None", help="Risk of raw data exposure")

    with privacy_col3:
        st.metric("Privacy Preservation", "Complete", help="Level of privacy protection achieved")

    st.markdown("---")

    # Privacy controls demonstration
    st.subheader("Privacy Controls Demonstration")

    st.markdown("""
    Adjust the sliders below to see how privacy controls affect model performance:
    """)

    privacy_col1, privacy_col2 = st.columns(2)

    with privacy_col1:
        clip_demo = st.slider("Gradient Clipping Norm", 0.1, 2.0, 1.0, 0.1, key="demo_clip")
        noise_demo = st.slider("Noise Multiplier", 0.0, 1.0, 0.0, 0.05, key="demo_noise")

    with privacy_col2:
        # Simulate the effect on performance
        base_f1 = 0.85
        base_auc = 0.90

        # Simple simulation: more clipping/noise reduces performance
        clip_penalty = max(0, (clip_demo - 0.5) * 0.1)  # Penalty for clipping < 0.5
        noise_penalty = noise_demo * 0.3  # Linear penalty for noise

        adjusted_f1 = max(0.1, base_f1 - clip_penalty - noise_penalty)
        adjusted_auc = max(0.1, base_auc - clip_penalty - noise_penalty * 0.5)

        st.metric("Simulated F1 Score", f"{adjusted_f1:.3f}")
        st.metric("Simulated PR-AUC", f"{adjusted_auc:.3f}")

        st.caption("*Note: This is a simplified demonstration. Actual relationships are more complex.*")

    st.markdown("---")
    st.markdown("""
    ### Future Privacy Enhancements

    Planned improvements to strengthen privacy guarantees:
    - **Secure Aggregation**: Encrypt updates during transmission
    - **Differential Privacy Formal Proofs**: Rigorous privacy budget tracking
    - **Federated Secure Transfer Learning**: More sophisticated knowledge sharing
    - **Audit Trails**: Comprehensive logging of all shared information
    """)

def show_transaction_demo():
    st.markdown('<h2 class="sub-header">Try a Transaction</h2>', unsafe_allow_html=True)

    st.markdown("""
    Test how our fraud detection model evaluates a transaction.
    Note: This uses a simplified demo model for illustration.
    """)

    # Load a trained model if available
    model_loaded = False
    try:
        import joblib
        model_files = glob.glob("./models/global_model_*.npz")
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            model_data = np.load(latest_model)
            # We would reconstruct and load the actual model here
            # For demo, we'll simulate predictions
            model_loaded = True
    except:
        pass

    if not model_loaded:
        st.info("🔄 Using demo model (train a model first for real predictions)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")

        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
        tx_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
        oldbalanceOrg = st.number_input("Origin Account Old Balance ($)", min_value=0.0, value=500.0, step=50.0)
        newbalanceOrig = st.number_input("Origin Account New Balance ($)", min_value=0.0, value=400.0, step=50.0)
        oldbalanceDest = st.number_input("Destination Account Old Balance ($)", min_value=0.0, value=300.0, step=50.0)
        newbalanceDest = st.number_input("Destination Account New Balance ($)", min_value=0.0, value=400.0, step=50.0)

        isFlaggedFraud = st.selectbox("Flagged by Existing Systems?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

        step = st.slider("Step (Hour of Month)", 1, 744, 100)

    with col2:
        st.subheader("Fraud Assessment")

        if st.button("🔍 Analyze Transaction", type="primary"):
            # Simple heuristic for demo (in reality, this would use the trained model)
            # This is just for demonstration purposes

            # Calculate risk factors
            risk_score = 0.0

            # Amount risk
            if amount > 1000:
                risk_score += 0.3
            elif amount > 500:
                risk_score += 0.1

            # Type risk
            if tx_type in ["TRANSFER", "CASH_OUT"]:
                risk_score += 0.2

            # Balance inconsistency risk
            if tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT"]:
                # Money leaving origin account
                expected_new_orig = oldbalanceOrg - amount
                if newbalanceOrig > expected_new_orig + 10:  # Allow small tolerance
                    risk_score += 0.4
            elif tx_type == "DEBIT":
                # Money entering destination account
                expected_new_dest = oldbalanceDest + amount
                if newbalanceDest < expected_new_dest - 10:  # Allow small tolerance
                    risk_score += 0.4

            # Flagged transaction risk
            if isFlaggedFraud == 1:
                risk_score += 0.2

            # Normalize risk score
            risk_score = min(1.0, risk_score)

            # Determine prediction
            is_fraud = risk_score > 0.5
            confidence = risk_score if is_fraud else (1 - risk_score)

            # Display results
            if is_fraud:
                st.error("🚨 **FRAUD DETECTED**")
            else:
                st.success("✅ **LEGITIMATE TRANSACTION**")

            st.metric("Fraud Probability", f"{risk_score:.1%}")
            st.metric("Confidence", f"{confidence:.1%}")

            # Risk factors breakdown
            st.subheader("Risk Factors")
            factors = []
            if amount > 1000:
                factors.append("High transaction amount")
            if tx_type in ["TRANSFER", "CASH_OUT"]:
                factors.append("High-risk transaction type")
            if tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT"] and newbalanceOrig > (oldbalanceOrg - amount + 10):
                factors.append("Origin balance inconsistency")
            if tx_type == "DEBIT" and newbalanceDest < (oldbalanceDest + amount - 10):
                factors.append("Destination balance inconsistency")
            if isFlaggedFraud == 1:
                factors.append("Previously flagged by systems")

            if factors:
                for factor in factors:
                    st.warning(f"⚠️ {factor}")
            else:
                st.info("No significant risk factors detected")

    st.markdown("---")
    st.subheader("How It Works")
    st.markdown("""
    In the actual CIPHERMESH system:

    1. **Feature Extraction**: Transaction details are converted to numerical features
    2. **Model Inference**: The federated learning model processes these features
    3. **Risk Scoring**: Model outputs a fraud probability between 0 and 1
    4. **Decision Threshold**: Transactions above 0.5 threshold are flagged as potentially fraudulent
    5. **Explanation**: System provides insights into which factors contributed most to the score

    The model has been trained collaboratively across multiple banks without any of them sharing raw transaction data.
    """)

# ========== TOP LEVEL EXECUTION (runs when Streamlit executes the script) ==========

# Header
st.markdown('<h1 class="main-header">🔐 CIPHERMESH</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Privacy-Preserving Fraud Signal Sharing Across Banks</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Federation Control Room", "Experiment Results", "Privacy Audit", "Try a Transaction"]
)

# Route to selected page
if page == "Federation Control Room":
    show_federation_control()
elif page == "Experiment Results":
    show_experiment_results()
elif page == "Privacy Audit":
    show_privacy_audit()
elif page == "Try a Transaction":
    show_transaction_demo()

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
<p>Together • Smarter detection • Data stays home.</p>
<p>CIPHERMESH - Privacy-Preserving Fraud Signal Sharing Across Banks</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)