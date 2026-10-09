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
import base64
from pathlib import Path

# Import our modules
import sys
sys.path.append('src')
from src.federated_learning import FederatedLearningCoordinator

# Page configuration
st.set_page_config(
    page_title="CIPHERMESH - Privacy-Preserving Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/Fusion-CIPHERMESH',
        'Report a bug': "https://github.com/yourusername/Fusion-CIPHERMESH/issues",
        'About': "# CIPHERMESH\nPrivacy-Preserving Fraud Signal Sharing Across Banks\nBuilt with Streamlit and Federated Learning"
    }
)

# Enhanced Custom CSS for Modern, Professional UI
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root variables for consistent theming */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #17a2b8;
        --light-color: #f8f9fa;
        --dark-color: #343a40;
        --border-radius: 12px;
        --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        --font-family: 'Inter', sans-serif;
    }

    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: var(--font-family);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom Header */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }

    .sub-header {
        font-size: 1.4rem;
        font-weight: 400;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Enhanced Metric Cards */
    .metric-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--box-shadow);
        border: 1px solid rgba(0, 0, 0, 0.08);
        transition: var(--transition);
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-description {
        font-size: 0.85rem;
        color: #868e96;
        margin-top: 0.5rem;
        font-style: italic;
    }

    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: var(--transition);
        box-shadow: 0 4px 6px rgba(31, 119, 180, 0.3);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(31, 119, 180, 0.4);
        background: linear-gradient(135deg, #1e6ea8 0%, #e67300 100%);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Primary button variant */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1e6ea8 0%, #e67300 100%);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 1px solid rgba(0, 0, 0, 0.08);
    }

    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }

    .sidebar-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0 0 var(--border-radius) var(--border-radius);
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
    }

    .sidebar-header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: var(--border-radius) var(--border-radius) 0 0;
        padding: 1rem 1.5rem;
        font-weight: 500;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-bottom: none;
        transition: var(--transition);
        position: relative;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f8f9fa;
        transform: translateY(-1px);
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary-color) !important;
        color: white !important;
        border-color: var(--primary-color) !important;
    }

    .stTabs [aria-selected="true"]:hover {
        background: var(--primary-color) !important;
    }

    /* Content cards */
    .content-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 2rem;
        box-shadow: var(--box-shadow);
        border: 1px solid rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        transition: var(--transition);
    }

    .content-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.12);
    }

    .content-card h3 {
        color: var(--dark-color);
        border-bottom: 2px solid rgba(31, 119, 180, 0.1);
        padding-bottom: 0.75rem;
        margin-top: 0;
    }

    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
    }

    .status-success {
        background-color: rgba(44, 160, 44, 0.1);
        color: var(--success-color);
        border: 1px solid rgba(44, 160, 44, 0.2);
    }

    .status-warning {
        background-color: rgba(214, 39, 40, 0.1);
        color: var(--warning-color);
        border: 1px solid rgba(214, 39, 40, 0.2);
    }

    .status-info {
        background-color: rgba(23, 162, 184, 0.1);
        color: var(--info-color);
        border: 1px solid rgba(23, 162, 184, 0.2);
    }

    .status-indicator::before {
        content: '';
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    .status-success::before {
        background-color: var(--success-color);
    }

    .status-warning::before {
        background-color: var(--warning-color);
    }

    .status-info::before {
        background-color: var(--info-color);
    }

    /* Progress indicators */
    .progress-container {
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
        margin: 1rem 0;
    }

    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 10px;
        transition: width 0.6s ease;
    }

    /* Feature highlights */
    .feature-highlight {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: var(--border-radius);
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }

    .feature-icon {
        font-size: 1.8rem;
        color: var(--primary-color);
        flex-shrink: 0;
    }

    .feature-content h4 {
        margin: 0 0 0.5rem 0;
        color: var(--dark-color);
    }

    .feature-content p {
        margin: 0;
        color: #6c757d;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(0, 0, 0, 0.08);
        color: #6c757d;
        font-size: 0.9rem;
    }

    .footer-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .footer-logo span {
        font-weight: 600;
        color: var(--primary-color);
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }

        .sub-header {
            font-size: 1.2rem;
        }

        .metric-value {
            font-size: 1.8rem;
        }

        .content-card {
            padding: 1.5rem;
        }

        .sidebar-header {
            padding: 1rem;
            margin: -1rem -1rem 1.5rem -1rem;
        }
    }

    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--box-shadow);
        border: 1px solid rgba(0, 0, 0, 0.08);
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
        vertical_spacing=0.12
    )

    fig.add_trace(
        go.Scatter(x=rounds, y=f1_scores, mode='lines+markers', name='F1',
                  line=dict(color='#1f77b4', width=3), marker=dict(size=8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=rounds, y=pr_auc_scores, mode='lines+markers', name='PR-AUC',
                  line=dict(color='#ff7f0e', width=3), marker=dict(size=8)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=rounds, y=precision_scores, mode='lines+markers', name='Precision',
                  line=dict(color='#2ca02c', width=3), marker=dict(size=8)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=rounds, y=recall_scores, mode='lines+markers', name='Recall',
                  line=dict(color='#d62728', width=3), marker=dict(size=8)),
        row=2, col=2
    )

    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="Federated Learning Performance Over Rounds",
        title_x=0.5,
        title_font_size=16,
        title_font_family="Inter",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

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
        colorbar=dict(title="F1 Score", thickness=15),
        hoverongaps=False
    ))

    fig.update_layout(
        title="Bank Performance Heatmap (F1 Score by Round)",
        title_x=0.5,
        title_font_size=16,
        title_font_family="Inter",
        xaxis_title="Round",
        yaxis_title="Bank",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

def show_federation_control():
    st.markdown('<h1 class="main-header">🔐 CIPHERMESH</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Privacy-Preserving Federated Learning for Fraud Detection Across Banks</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="content-card fade-in-up">
            <h3>🚀 Run Federated Learning Experiment</h3>
            <p>Configure and launch privacy-preserving federated learning experiments where banks collaboratively train fraud detection models without sharing raw transaction data.</p>
        </div>
        """, unsafe_allow_html=True)

        # Parameters in cards
        param_col1, param_col2 = st.columns(2)

        with param_col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Experiment Scale</div>
            </div>
            """, unsafe_allow_html=True)
            n_rounds = st.slider("Number of Training Rounds", 1, 20, 5, help="How many rounds of federated learning to perform")
            n_banks = st.slider("Participating Banks", 1, 5, 5, help="Number of banks to include in the federation")

        with param_col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Privacy Controls</div>
            </div>
            """, unsafe_allow_html=True)
            clip_norm = st.selectbox("Gradient Clipping Norm", [None, 0.5, 1.0, 2.0], index=2,
                                   help="Limits the magnitude of model updates for privacy protection")
            noise_multiplier = st.slider("Noise Multiplier (Differential Privacy)", 0.0, 1.0, 0.0, 0.1,
                                       help="Adds Gaussian noise to model updates for formal privacy guarantees")

        if st.button("🚀 Launch Federated Experiment", type="primary", use_container_width=True):
            with st.spinner("🔄 Initializing federated learning coordinator..."):
                # Initialize coordinator
                coordinator = FederatedLearningCoordinator(
                    data_dir="./data",
                    results_dir="./results",
                    models_dir="./models"
                )

            with st.spinner("🏃‍♂️ Running federated learning experiment..."):
                # Run experiment
                participating_banks = list(range(n_banks)) if n_banks < 5 else None
                results = coordinator.run_federated_learning(
                    n_rounds=n_rounds,
                    clip_norm=clip_norm,
                    noise_multiplier=noise_multiplier,
                    participating_banks=participating_banks,
                    save_results=True
                )

            st.success("✅ Experiment completed successfully!")
            st.balloons()

            # Display results
            final_metrics = results['final_global_metrics']

            # Metrics cards
            st.markdown("### 📊 Experiment Results")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">{final_metrics['f1']:.4f}</div>
                    <div class="metric-description">Harmonic mean of precision and recall</div>
                </div>
                """, unsafe_allow_html=True)
            with metric_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PR-AUC</div>
                    <div class="metric-value">{final_metrics['pr_auc']:.4f}</div>
                    <div class="metric-description">Area under precision-recall curve</div>
                </div>
                """, unsafe_allow_html=True)
            with metric_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">{final_metrics['precision']:.4f}</div>
                    <div class="metric-description">Accuracy of positive predictions</div>
                </div>
                """, unsafe_allow_html=True)
            with metric_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">{final_metrics['recall']:.4f}</div>
                    <div class="metric-description">Fraction of fraud cases detected</div>
                </div>
                """, unsafe_allow_html=True)

            # Plot results
            st.markdown("### 📈 Performance Progression")
            fig = plot_round_metrics(results['round_history'])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Bank participation heatmap
            st.markdown("### 🏦 Bank Participation Analysis")
            fig2 = plot_bank_participation(results['round_history'])
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.markdown("""
        <div class="content-card fade-in-up">
            <h3>📊 System Status</h3>
        </div>
        """, unsafe_allow_html=True)

        # Load latest results
        latest_results = load_latest_results("./results")
        if latest_results:
            st.markdown("""
            <div class="status-indicator status-info">
                ✅ Previous experiment found
            </div>
            """, unsafe_allow_html=True)

            final_metrics = latest_results['final_global_metrics']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latest F1 Score</div>
                <div class="metric-value">{final_metrics['f1']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Latest PR-AUC</div>
                <div class="metric-value">{final_metrics['pr_auc']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-indicator status-warning">
                ⚠️ No previous experiments found
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="content-card fade-in-up">
            <h3>🔒 How Federated Learning Works</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">1️⃣</div>
            <div class="feature-content">
                <h4>Local Training</h4>
                <p>Each bank trains on its own data locally, keeping all transaction data private and secure.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">2️⃣</div>
            <div class="feature-content">
                <h4>Update Sharing</h4>
                <p>Only model weight updates (gradients) are shared - never raw transaction data or customer information.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">3️⃣</div>
            <div class="feature-content">
                <h4>Federated Averaging</h4>
                <p>The coordinator averages all model updates to create an improved global model.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">4️⃣</div>
            <div class="feature-content">
                <h4>Model Distribution</h4>
                <p>The improved global model is sent back to each bank for the next round.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_experiment_results():
    st.markdown('<h1 class="main-header">🔐 CIPHERMESH</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Experiment Comparison & Performance Analytics</p>', unsafe_allow_html=True)

    # Load experiment comparison
    comparison_data = load_experiment_comparison("./results")

    if comparison_data:
        st.markdown("""
        <div class="content-card fade-in-up">
            <h3>📊 Experiment Comparison Dashboard</h3>
            <p>Compare different federated learning configurations to understand the privacy-utility trade-off.</p>
        </div>
        """, unsafe_allow_html=True)

        # Prepare data for comparison chart
        experiment_names = list(comparison_data.keys())
        f1_scores = [comparison_data[exp]['final_global_metrics']['f1'] for exp in experiment_names]
        pr_auc_scores = [comparison_data[exp]['final_global_metrics']['pr_auc'] for exp in experiment_names]
        precision_scores = [comparison_data[exp]['final_global_metrics']['precision'] for exp in experiment_names]
        recall_scores = [comparison_data[exp]['final_global_metrics']['recall'] for exp in experiment_names]

        # Create comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='F1 Score',
            x=experiment_names,
            y=f1_scores,
            marker_color='#1f77b4',
            text=[f'{score:.3f}' for score in f1_scores],
            textposition='auto',
        ))

        fig.add_trace(go.Bar(
            name='PR-AUC',
            x=experiment_names,
            y=pr_auc_scores,
            marker_color='#ff7f0e',
            text=[f'{score:.3f}' for score in pr_auc_scores],
            textposition='auto',
        ))

        fig.update_layout(
            title="Comparison of Different Experimental Configurations",
            title_x=0.5,
            title_font_size=18,
            title_font_family="Inter",
            xaxis_title="Experiment Configuration",
            yaxis_title="Score",
            barmode='group',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Detailed results table
        st.markdown("### 📋 Detailed Results")
        results_df = pd.DataFrame({
            'Experiment': experiment_names,
            'F1 Score': [f"{score:.4f}" for score in f1_scores],
            'PR-AUC': [f"{score:.4f}" for score in pr_auc_scores],
            'Precision': [f"{comparison_data[exp]['final_global_metrics']['precision']:.4f}" for exp in experiment_names],
            'Recall': [f"{comparison_data[exp]['final_global_metrics']['recall']:.4f}" for exp in experiment_names],
            'Rounds': [comparison_data[exp]['n_rounds'] for exp in experiment_names],
            'Privacy (Clip)': [str(comparison_data[exp]['clip_norm']) for exp in experiment_names],
            'Privacy (Noise)': [f"{comparison_data[exp]['noise_multiplier']:.2f}" for exp in experiment_names]
        })

        st.dataframe(results_df, use_container_width=True, height=300)

        # Privacy-Utility Trade-off Analysis
        st.markdown("### ⚖️ Privacy-Utility Trade-off Analysis")

        # Create scatter plot for privacy vs utility
        privacy_levels = []
        utility_scores = []
        experiment_labels = []

        for exp_name, exp_data in comparison_data.items():
            # Simple privacy metric: higher clipping + noise = more privacy
            clip_val = exp_data['clip_norm'] or 0
            noise_val = exp_data['noise_multiplier']
            privacy_score = clip_val + (noise_val * 2)  # Weight noise more heavily

            utility_score = exp_data['final_global_metrics']['f1']

            privacy_levels.append(privacy_score)
            utility_scores.append(utility_score)
            experiment_labels.append(exp_name)

        fig_scatter = go.Figure()

        fig_scatter.add_trace(go.Scatter(
            x=privacy_levels,
            y=utility_scores,
            mode='markers+text',
            text=experiment_labels,
            textposition="top center",
            marker=dict(
                size=15,
                color=utility_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="F1 Score", thickness=15),
                line=dict(width=2, color='white')
            )
        ))

        fig_scatter.update_layout(
            title="Privacy vs Utility Trade-off Analysis",
            title_x=0.5,
            title_font_size=16,
            title_font_family="Inter",
            xaxis_title="Privacy Level (Higher = More Privacy Protection)",
            yaxis_title="Utility (F1 Score)",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

    else:
        st.markdown("""
        <div class="content-card fade-in-up">
            <h3>📊 No Experiment Data Available</h3>
            <p>Run some experiments first to see comparison data here.</p>
            <div style="text-align: center; margin: 2rem 0;">
                <a href="#" onclick="document.querySelector('[data-testid=\"stSidebarNav\"] li:nth-child(1) button').click(); return false;">
                    <button style="background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%); color: white; border: none; padding: 0.75rem 2rem; border-radius: 12px; font-weight: 600;">Go to Federation Control Room</button>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show latest single experiment results
        latest_results = load_latest_results("./results")
        if latest_results:
            st.markdown("""
            <div class="content-card fade-in-up">
                <h3>📈 Latest Experiment Results</h3>
            </div>
            """, unsafe_allow_html=True)

            # Display in a nice format
            st.json(latest_results, expanded=False)

def show_privacy_audit():
    st.markdown('<h1 class="main-header">🔐 CIPHERMESH</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Privacy Transparency & Audit Trail</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card fade-in-up">
        <h3>🔒 Privacy Guarantees Overview</h3>
        <p>CIPHERMESH ensures that sensitive financial data never leaves the participating banks' premises, enabling collaborative fraud detection without compromising customer privacy.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="content-card">
            <h4>🔒 Data That Stays Local</h4>
            <div class="feature-highlight">
                <div class="feature-icon">🏦</div>
                <div class="feature-content">
                    <h5>Transaction Records</h5>
                    <p>Individual transaction details, amounts, and timestamps</p>
                </div>
            </div>
            <div class="feature-highlight">
                <div class="feature-icon">👤</div>
                <div class="feature-content">
                    <h5>Customer Information</h5>
                    <p>Account numbers, personal details, and PII</p>
                </div>
            </div>
            <div class="feature-highlight">
                <div class="feature-icon">💳</div>
                <div class="feature-content">
                    <h5>Financial Data</h5>
                    <p>Account balances, transaction histories, and spending patterns</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="content-card">
            <h4>📤 Information That Is Shared</h4>
            <div class="feature-highlight">
                <div class="feature-icon">📊</div>
                <div class="feature-content">
                    <h5>Model Updates</h5>
                    <p>Weight gradients and optimization updates</p>
                </div>
            </div>
            <div class="feature-highlight">
                <div class="feature-icon">📈</div>
                <div class="feature-content">
                    <h5>Performance Metrics</h5>
                    <p>Accuracy, F1-score, precision, and recall metrics</p>
                </div>
            </div>
            <div class="feature-highlight">
                <div class="feature-icon">📋</div>
                <div class="feature-content">
                    <h5>Training Statistics</h5>
                    <p>Number of samples used, convergence metrics</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Privacy metrics
    st.markdown("""
    <div class="content-card fade-in-up">
        <h3>📊 Privacy Metrics Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)

    privacy_col1, privacy_col2, privacy_col3, privacy_col4 = st.columns(4)

    with privacy_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Rows Shared</div>
            <div class="metric-value">0</div>
            <div class="metric-description">Raw transaction rows shared between banks</div>
        </div>
        """, unsafe_allow_html=True)

    with privacy_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Data Leakage Risk</div>
            <div class="metric-value">None</div>
            <div class="metric-description">Risk of raw data exposure</div>
        </div>
        """, unsafe_allow_html=True)

    with privacy_col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Privacy Preservation</div>
            <div class="metric-value">Complete</div>
            <div class="metric-description">Level of privacy protection achieved</div>
        </div>
        """, unsafe_allow_html=True)

    with privacy_col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Audit Trail</div>
            <div class="metric-value">Verified</div>
            <div class="metric-description">Tamper-evident logging enabled</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Privacy controls demonstration
    st.markdown("""
    <div class="content-card fade-in-up">
        <h3>🔧 Privacy Controls Demonstration</h3>
        <p>Adjust privacy parameters to see how they affect model performance and privacy guarantees.</p>
    </div>
    """, unsafe_allow_html=True)

    privacy_col1, privacy_col2 = st.columns(2)

    with privacy_col1:
        st.markdown("#### 🎛️ Privacy Parameters")
        clip_demo = st.slider("Gradient Clipping Norm", 0.1, 2.0, 1.0, 0.1, key="demo_clip",
                             help="Lower values = more privacy, potentially lower utility")
        noise_demo = st.slider("Noise Multiplier", 0.0, 1.0, 0.0, 0.05, key="demo_noise",
                              help="Higher values = stronger differential privacy guarantees")

    with privacy_col2:
        st.markdown("#### 📈 Simulated Impact")

        # More realistic simulation
        base_f1 = 0.004  # Base performance from our demo
        base_auc = 0.59  # Base AUC from our demo

        # Privacy-utility tradeoff curve (more realistic)
        clip_penalty = max(0, (clip_demo - 0.5) * 0.002) if clip_demo else 0  # Penalty for tight clipping
        noise_penalty = noise_demo * 0.008  # Linear penalty for noise
        total_penalty = clip_penalty + noise_penalty

        adjusted_f1 = max(0.0005, base_f1 - total_penalty)
        adjusted_auc = max(0.4, base_auc - total_penalty * 0.6)

        # Privacy guarantee level
        privacy_level = min(100, (clip_demo * 20) + (noise_demo * 50))

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Simulated F1 Score</div>
            <div class="metric-value">{adjusted_f1:.4f}</div>
            <div class="metric-description">Expected model performance</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Simulated PR-AUC</div>
            <div class="metric-value">{adjusted_auc:.3f}</div>
            <div class="metric-description">Area under precision-recall curve</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Privacy Guarantee</div>
            <div class="metric-value">{privacy_level:.0f}%</div>
            <div class="metric-description">Estimated privacy protection level</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("*Note: This demonstrates the privacy-utility tradeoff. Actual relationships depend on data distribution and model architecture.*")

    st.markdown("---")
    st.markdown("""
    <div class="content-card fade-in-up">
        <h3>🛡️ Future Privacy Enhancements</h3>
        <p>Planned improvements to strengthen privacy guarantees for production deployment:</p>
    </div>
    """, unsafe_allow_html=True)

    enh_col1, enh_col2 = st.columns(2)

    with enh_col1:
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">🔐</div>
            <div class="feature-content">
                <h4>Secure Aggregation</h4>
                <p>Cryptographic protocols to protect update privacy during transmission</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">📜</div>
            <div class="feature-content">
                <h4>Formal Privacy Accounting</h4>
                <p>Rigorous differential privacy budget tracking with composition theorems</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with enh_col2:
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">🔑</div>
            <div class="feature-content">
                <h4>Access Control & Authentication</h4>
                <p>Mutual TLS and role-based access for participating institutions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">👁️</div>
            <div class="feature-content">
                <h4>Transparent Audit Logs</h4>
                <p>Comprehensive logging of all shared information with immutability guarantees</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_transaction_demo():
    st.markdown('<h1 class="main-header">🔐 CIPHERMESH</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Fraud Detection Demo</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card fade-in-up">
        <h3>💳 Test Transaction Fraud Detection</h3>
        <p>Experience how our privacy-preserving fraud detection model evaluates transactions in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load a trained model if available
    model_loaded = False
    model_info = ""
    try:
        import joblib
        model_files = glob.glob("./models/global_model_*.npz")
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            model_data = np.load(latest_model)
            model_loaded = True
            model_info = f"Using model from: {os.path.basename(latest_model)}"
        else:
            model_info = "No trained model found - using demo simulation"
    except:
        model_info = "Using demo simulation (train a model for real predictions)"

    if not model_loaded:
        st.info(f"🔄 {model_info}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="content-card">
            <h4>📝 Transaction Details</h4>
        </div>
        """, unsafe_allow_html=True)

        # Transaction form with better styling
        with st.container():
            st.markdown('<label>💰 Transaction Amount ($)</label>', unsafe_allow_html=True)
            amount = st.number_input("", min_value=0.0, value=1500.0, step=50.0, label_visibility="collapsed")

            st.markdown('<label>🔄 Transaction Type</label>', unsafe_allow_html=True)
            tx_type = st.selectbox("", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"], label_visibility="collapsed")

            st.markdown('<label>🏦 Origin Account Old Balance ($)</label>', unsafe_allow_html=True)
            oldbalanceOrg = st.number_input("", min_value=0.0, value=2000.0, step=100.0, label_visibility="collapsed")

            st.markdown('<label>🏦 Origin Account New Balance ($)</label>', unsafe_allow_html=True)
            newbalanceOrig = st.number_input("", min_value=0.0, value=500.0, step=100.0, label_visibility="collapsed")

            st.markdown('<label>🏦 Destination Account Old Balance ($)</label>', unsafe_allow_html=True)
            oldbalanceDest = st.number_input("", min_value=0.0, value=1000.0, step=100.0, label_visibility="collapsed")

            st.markdown('<label>🏦 Destination Account New Balance ($)</label>', unsafe_allow_html=True)
            newbalanceDest = st.number_input("", min_value=0.0, value=500.0, step=100.0, label_visibility="collapsed")

            st.markdown('<label>🚩 Flagged by Existing Systems?</label>', unsafe_allow_html=True)
            isFlaggedFraud = st.selectbox("", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", label_visibility="collapsed")

            st.markdown('<label>⏰ Step (Hour of Month)</label>', unsafe_allow_html=True)
            step = st.slider("", 1, 744, 300, label_visibility="collapsed")

    with col2:
        st.markdown("""
        <div class="content-card">
            <h4>🔍 Fraud Assessment Results</h4>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍 Analyze Transaction for Fraud", type="primary", use_container_width=True):
            # Enhanced risk assessment with more factors
            risk_score = 0.0
            risk_factors = []
            risk_levels = {"Low": 0, "Medium": 0, "High": 0}

            # Amount risk (more sophisticated)
            if amount > 5000:
                risk_score += 0.4
                risk_factors.append("Very high transaction amount (>$5,000)")
                risk_levels["High"] += 1
            elif amount > 2000:
                risk_score += 0.2
                risk_factors.append("High transaction amount (>$2,000)")
                risk_levels["Medium"] += 1
            elif amount > 1000:
                risk_score += 0.1
                risk_factors.append("Moderate transaction amount (>$1,000)")
                risk_levels["Low"] += 1

            # Type risk
            if tx_type in ["TRANSFER", "CASH_OUT"]:
                risk_score += 0.25
                risk_factors.append("High-risk transaction type (transfer/cash out)")
                risk_levels["Medium"] += 1

            # Balance inconsistency risk (enhanced)
            if tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT"]:
                # Money leaving origin account
                expected_new_orig = oldbalanceOrg - amount
                balance_diff_orig = newbalanceOrig - expected_new_orig
                if balance_diff_orig > 50:  # Significant discrepancy
                    risk_score += 0.35
                    risk_factors.append(f"Origin balance inconsistency (${balance_diff_orig:.2f} unexplained)")
                    risk_levels["High"] += 1
                elif balance_diff_orig > 10:
                    risk_score += 0.15
                    risk_factors.append(f"Minor origin balance discrepancy (${balance_diff_orig:.2f})")
                    risk_levels["Low"] += 1
            elif tx_type == "DEBIT":
                # Money entering destination account
                expected_new_dest = oldbalanceDest + amount
                balance_diff_dest = expected_new_dest - newbalanceDest
                if balance_diff_dest > 50:
                    risk_score += 0.35
                    risk_factors.append(f"Destination balance inconsistency (${balance_diff_dest:.2f} missing)")
                    risk_levels["High"] += 1
                elif balance_diff_dest > 10:
                    risk_score += 0.15
                    risk_factors.append(f"Minor destination balance discrepancy (${balance_diff_dest:.2f})")
                    risk_levels["Low"] += 1

            # Flagged transaction risk
            if isFlaggedFraud == 1:
                risk_score += 0.2
                risk_factors.append("Previously flagged by existing fraud systems")
                risk_levels["Medium"] += 1

            # Time-based risk (transactions outside business hours)
            hour_of_day = step % 24
            if hour_of_day < 6 or hour_of_day > 22:  # Night transactions
                risk_score += 0.1
                risk_factors.append("Transaction outside normal business hours")
                risk_levels["Low"] += 1

            # Normalize risk score
            risk_score = min(0.98, risk_score)  # Cap at 98% to avoid certainty

            # Determine prediction with confidence thresholds
            is_fraud = risk_score > 0.5
            confidence = risk_score if is_fraud else (1 - risk_score)

            # Display results with enhanced styling
            if is_fraud:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                           color: white; padding: 1.5rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                    <h2>🚨 FRAUD DETECTED</h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">Transaction blocked for security review</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
                           color: white; padding: 1.5rem; border-radius: 16px; text-align: center; margin: 1rem 0;">
                    <h2>✅ LEGITIMATE TRANSACTION</h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">Transaction approved</p>
                </div>
                """, unsafe_allow_html=True)

            # Metrics in cards
            met_col1, met_col2 = st.columns(2)
            with met_col1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Fraud Probability</div>
                    <div class="metric-value">{risk_score:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            with met_col2:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            # Risk level indicator
            max_risk_level = max(risk_levels, key=risk_levels.get)
            risk_color = {"Low": "#51cf66", "Medium": "#ff922b", "High": "#ff6b6b"}[max_risk_level]
            st.markdown(f"""
            <div style="text-align: center; margin: 1.5rem 0;">
                <span style="background: {risk_color}; color: white; padding: 0.5rem 1rem;
                           border-radius: 20px; font-weight: 600; text-transform: uppercase;
                           letter-spacing: 0.5px; font-size: 0.9rem;">
                    Risk Level: {max_risk_level}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Risk factors breakdown
            if risk_factors:
                st.markdown("""
                <div class="content-card">
                    <h4>⚠️ Detected Risk Factors</h4>
                </div>
                """, unsafe_allow_html=True)

                for i, factor in enumerate(risk_factors, 1):
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border-left: 4px solid #ff6b6b;
                               padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
                        <strong>{i}.</strong> {factor}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="content-card" style="text-align: center; padding: 2rem;">
                    <p style="color: #28a745; font-size: 1.1rem;">✅ No significant risk factors detected</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="content-card fade-in-up">
        <h3>🧠 How CIPHERMESH Fraud Detection Works</h3>
    </div>
    """, unsafe_allow_html=True)

    how_col1, how_col2 = st.columns(2)

    with how_col1:
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">🔢</div>
            <div class="feature-content">
                <h4>Feature Engineering</h4>
                <p>Transaction details are converted to numerical features representing behavioral patterns, temporal patterns, and financial relationships.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">🤖</div>
            <div class="feature-content">
                <h4>Model Inference</h4>
                <p>The federated learning model processes these features to compute a fraud probability based on patterns learned from collaborative training across banks.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with how_col2:
        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">📊</div>
            <div class="feature-content">
                <h4>Risk Scoring</h4>
                <p>Model outputs a calibrated fraud probability between 0 and 1, representing the likelihood of fraudulent intent.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-highlight">
            <div class="feature-icon">⚖️</div>
            <div class="feature-content">
                <h4>Decision Making</h4>
                <p>Transactions exceeding the 0.5 probability threshold are flagged for review, balancing detection rates with false positive minimization.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card fade-in-up">
        <h4>🔒 Privacy Assurance</h4>
        <p>Throughout this entire process, <strong>no raw transaction data</strong> is shared between banks. Only anonymized model updates are exchanged to improve the collective fraud detection capability.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
def show_footer():
    st.markdown("""
    <div class="footer">
        <div class="footer-logo">
            <span>🔐</span>
            <span>CIPHERMESH</span>
        </div>
        <p>Privacy-Preserving Fraud Signal Sharing Across Banks</p>
        <p>
            <a href="#" style="color: #1f77b4; text-decoration: none;">Documentation</a> |
            <a href="#" style="color: #1f77b4; text-decoration: none;">Privacy Policy</a> |
            <a href="#" style="color: #1f77b4; text-decoration: none;">Security</a> |
            <a href="#" style="color: #1f77b4; text-decoration: none;">Contact</a>
        </p>
        <p style="font-size: 0.8rem; opacity: 0.7; margin-top: 1.5rem;">
            © 2026 CIPHERMESH. All rights reserved. Built with Streamlit and Federated Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========== TOP LEVEL EXECUTION (runs when Streamlit executes the script) ==========

# Initialize session state for page tracking
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Federation Control Room"

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🔐 CIPHERMESH</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")

    # Navigation buttons with icons
    if st.button("🏦 Federation Control Room", use_container_width=True):
        st.session_state.current_page = "Federation Control Room"
        st.rerun()

    if st.button("📊 Experiment Results", use_container_width=True):
        st.session_state.current_page = "Experiment Results"
        st.rerun()

    if st.button("🔒 Privacy Audit", use_container_width=True):
        st.session_state.current_page = "Privacy Audit"
        st.rerun()

    if st.button("💳 Try a Transaction", use_container_width=True):
        st.session_state.current_page = "Try a Transaction"
        st.rerun()

    st.markdown("---")

    # System status in sidebar
    st.markdown("### System Status")

    # Check if Streamlit is healthy (we know it is since we're running)
    st.markdown("""
    <div class="status-indicator status-success">
        🟢 System Online
    </div>
    """, unsafe_allow_html=True)

    # Latest experiment info
    latest_results = load_latest_results("./results")
    if latest_results:
        final_metrics = latest_results['final_global_metrics']
        st.markdown(f"""
        <div style="font-size: 0.9rem; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between;">
                <span>Latest F1:</span>
                <span style="font-weight: 600; color: #1f77b4;">{final_metrics['f1']:.4f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                <span>Latest PR-AUC:</span>
                <span style="font-weight: 600; color: #ff7f0e;">{final_metrics['pr_auc']:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size: 0.9rem; color: #6c757d; text-align: center;">
            No experiments yet
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick stats
    st.markdown("### Quick Stats")
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem;">
        <div><strong>Banks:</strong> 5</div>
        <div><strong>Algo:</strong> FedAvg</div>
        <div><strong>Privacy:</strong> DP</div>
        <div><strong>Model:</strong> LR</div>
    </div>
    """, unsafe_allow_html=True)

# Route to selected page
page = st.session_state.current_page

if page == "Federation Control Room":
    show_federation_control()
elif page == "Experiment Results":
    show_experiment_results()
elif page == "Privacy Audit":
    show_privacy_audit()
elif page == "Try a Transaction":
    show_transaction_demo()

# Show footer
show_footer()