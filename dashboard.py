"""
Monitoring Dashboard for Predictive Maintenance ML Pipeline
===========================================================
A Streamlit-based dashboard for monitoring model performance, drift detection,
fairness metrics, and system health.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys

# Set page config
st.set_page_config(
    page_title="Predictive Maintenance - Monitoring Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BASE_DIR = Path(__file__).parent
ARTIFACTS = BASE_DIR / "artifacts"

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_json(file_path):
    """Load JSON file with caching"""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {e}")
        return {}


@st.cache_data
def load_csv(file_path):
    """Load CSV file with caching"""
    try:
        if file_path.exists():
            return pd.read_csv(file_path)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {e}")
        return pd.DataFrame()


def get_status_color(value, thresholds):
    """Get status color based on value and thresholds"""
    if value <= thresholds['good']:
        return 'status-good'
    elif value <= thresholds['warning']:
        return 'status-warning'
    else:
        return 'status-error'


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">🔧 Predictive Maintenance - Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        st.markdown("---")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📁 Data Sources")
        st.info("Loading data from `artifacts/` directory")
        
        # Check artifact availability
        artifacts_status = {
            "Model Metrics": (ARTIFACTS / "model_metrics.json").exists(),
            "Fairness Report": (ARTIFACTS / "fairness_report.json").exists(),
            "Drift Report": (ARTIFACTS / "drift_report.csv").exists(),
            "LIME Metadata": (ARTIFACTS / "lime_meta.json").exists(),
            "SHAP Metadata": (ARTIFACTS / "shap_meta.json").exists(),
        }
        
        for artifact, available in artifacts_status.items():
            status = "✅" if available else "❌"
            st.text(f"{status} {artifact}")
        
        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    model_metrics = load_json(ARTIFACTS / "model_metrics.json")
    fairness_data = load_json(ARTIFACTS / "fairness_report.json")
    drift_df = load_csv(ARTIFACTS / "drift_report.csv")
    lime_meta = load_json(ARTIFACTS / "lime_meta.json")
    shap_meta = load_json(ARTIFACTS / "shap_meta.json")
    
    # Check if data is available
    if not model_metrics and not fairness_data:
        st.warning("⚠️ No model artifacts found. Please run the ML pipeline first: `python run_pipeline.py`")
        st.info("The dashboard will display data once artifacts are generated.")
        return
    
    # ========================================================================
    # SECTION 1: MODEL PERFORMANCE OVERVIEW
    # ========================================================================
    st.header("📈 Model Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Accuracy metric
    accuracy = model_metrics.get('accuracy', 0) if model_metrics else 0
    with col1:
        st.metric(
            label="Test Accuracy",
            value=f"{accuracy:.2%}",
            delta=None
        )
    
    # Precision metric
    precision = fairness_data.get('overall_precision', 0) if fairness_data else 0
    with col2:
        st.metric(
            label="Precision",
            value=f"{precision:.2%}",
            delta=None
        )
    
    # Recall metric
    recall = fairness_data.get('overall_recall', 0) if fairness_data else 0
    with col3:
        st.metric(
            label="Recall",
            value=f"{recall:.2%}",
            delta=None
        )
    
    # F1 Score metric
    f1 = fairness_data.get('overall_f1', 0) if fairness_data else 0
    with col4:
        st.metric(
            label="F1 Score",
            value=f"{f1:.2%}",
            delta=None
        )
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 2: DRIFT DETECTION
    # ========================================================================
    st.header("🔍 Data Drift Monitoring")
    
    if not drift_df.empty and 'drift_detected' in drift_df.columns:
        # Drift summary
        drift_count = drift_df['drift_detected'].sum()
        total_features = len(drift_df)
        drift_rate = (drift_count / total_features * 100) if total_features > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Features with Drift",
                value=f"{drift_count}/{total_features}",
                delta=f"{drift_rate:.1f}%"
            )
        
        with col2:
            status_class = get_status_color(drift_rate, {'good': 10, 'warning': 30})
            st.markdown(f"**Drift Rate:** <span class='{status_class}'>{drift_rate:.1f}%</span>", unsafe_allow_html=True)
        
        # Drift visualization
        if drift_count > 0:
            st.subheader("Drift Detection by Feature")
            
            # Create bar chart
            drift_plot_df = drift_df[['feature', 'drift_detected', 'p_value']].copy()
            drift_plot_df['Status'] = drift_plot_df['drift_detected'].map({True: 'Drift Detected', False: 'No Drift'})
            
            fig = px.bar(
                drift_plot_df,
                x='feature',
                y='p_value',
                color='Status',
                color_discrete_map={'Drift Detected': '#dc3545', 'No Drift': '#28a745'},
                labels={'p_value': 'P-Value', 'feature': 'Feature'},
                title="Drift Detection Results (Lower p-value = More Drift)")
            fig.add_hline(y=0.05, line_dash="dash", line_color="orange", annotation_text="Significance Threshold (0.05)")
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Drift details table
            with st.expander("📋 View Detailed Drift Report"):
                st.dataframe(drift_df[['feature', 'drift_detected', 'p_value', 'train_mean', 'test_mean']], use_container_width=True)
        else:
            st.success("✅ No drift detected across all features!")
    else:
        st.info("ℹ️ Drift detection data not available. Run the ML pipeline to generate drift reports.")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 3: FAIRNESS ANALYSIS
    # ========================================================================
    st.header("⚖️ Fairness Analysis by Machine Type")
    
    if fairness_data and 'by_type' in fairness_data:
        # Bias status
        bias_detected = fairness_data.get('bias_detected', False)
        col1, col2 = st.columns(2)
        
        with col1:
            if bias_detected:
                st.error("⚠️ Bias Detected")
            else:
                st.success("✅ No Major Bias Detected")
        
        with col2:
            demographic_parity = fairness_data.get('demographic_parity', 0)
            st.metric("Demographic Parity", f"{demographic_parity:.4f}")
        
        # Performance by type
        st.subheader("Performance Metrics by Machine Type")
        
        by_type = fairness_data.get('by_type', {})
        if by_type:
            # Create comparison dataframe
            metrics_df = pd.DataFrame({
                'Type': [],
                'Accuracy': [],
                'Precision': [],
                'Recall': [],
                'F1 Score': [],
                'FNR': []
            })
            
            for machine_type, metrics in by_type.items():
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Type': [machine_type],
                    'Accuracy': [metrics.get('accuracy', 0)],
                    'Precision': [metrics.get('precision', 0)],
                    'Recall': [metrics.get('recall', 0)],
                    'F1 Score': [metrics.get('f1_score', 0)],
                    'FNR': [metrics.get('false_negative_rate', 0)]
                })], ignore_index=True)
            
            # Display metrics table
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig = px.bar(
                    metrics_df,
                    x='Type',
                    y='Accuracy',
                    title="Accuracy by Machine Type",
                    color='Type',
                    color_discrete_map={'L': '#1f77b4', 'M': '#ff7f0e', 'H': '#2ca02c'}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # FNR comparison (critical for predictive maintenance)
                fig = px.bar(
                    metrics_df,
                    x='Type',
                    y='FNR',
                    title="False Negative Rate by Machine Type (Lower is Better)",
                    color='Type',
                    color_discrete_map={'L': '#1f77b4', 'M': '#ff7f0e', 'H': '#2ca02c'}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # FNR Analysis (if available)
            if 'fnr_analysis' in fairness_data:
                st.subheader("⚠️ FNR Disparity Analysis")
                fnr_info = fairness_data['fnr_analysis']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min FNR", f"{fnr_info.get('min_fnr', 0):.2%}")
                with col2:
                    st.metric("Max FNR", f"{fnr_info.get('max_fnr', 0):.2%}")
                with col3:
                    st.metric("FNR Difference", f"{fnr_info.get('fnr_difference', 0):.2%}")
                
                st.warning(f"FNR Ratio: {fnr_info.get('fnr_ratio', 1):.2f}x - {fnr_info.get('warning', '')}")
    else:
        st.info("ℹ️ Fairness analysis data not available.")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 4: FEATURE IMPORTANCE
    # ========================================================================
    st.header("🎯 Feature Importance")
    
    # Model feature importance
    if model_metrics and 'feature_importance' in model_metrics:
        st.subheader("Model Feature Importance")
        feature_importance = model_metrics['feature_importance']
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=False)
        
        # Visualization
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top Features by Importance",
            labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        with st.expander("📋 View All Feature Importances"):
            st.dataframe(importance_df, use_container_width=True)
    
    # SHAP feature importance (if available)
    if shap_meta and shap_meta.get('global_importance'):
        st.subheader("SHAP Global Feature Importance")
        shap_importance = shap_meta['global_importance']
        
        # Create dataframe
        shap_df = pd.DataFrame({
            'Feature': list(shap_importance.keys()),
            'SHAP Importance': list(shap_importance.values())
        }).sort_values('SHAP Importance', ascending=False)
        
        # Visualization
        fig = px.bar(
            shap_df,
            x='SHAP Importance',
            y='Feature',
            orientation='h',
            title="SHAP Feature Importance",
            labels={'SHAP Importance': 'Mean |SHAP Value|', 'Feature': 'Feature Name'},
            color='SHAP Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 5: SYSTEM HEALTH
    # ========================================================================
    st.header("💚 System Health Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Model status
    model_exists = (ARTIFACTS / "rf_model.pkl").exists()
    with col1:
        if model_exists:
            st.success("✅ Model Available")
        else:
            st.error("❌ Model Missing")
    
    # Explainability status
    lime_available = lime_meta and lime_meta.get('explanations')
    shap_available = shap_meta and shap_meta.get('shap_available', False)
    with col2:
        if lime_available and shap_available:
            st.success("✅ LIME + SHAP")
        elif lime_available or shap_available:
            st.warning("⚠️ Partial Explainability")
        else:
            st.error("❌ No Explainability")
    
    # Drift monitoring status
    drift_available = not drift_df.empty
    with col3:
        if drift_available:
            if drift_count == 0:
                st.success("✅ No Drift")
            else:
                st.warning(f"⚠️ {drift_count} Features with Drift")
        else:
            st.info("ℹ️ Drift Data Unavailable")
    
    # Fairness status
    fairness_available = bool(fairness_data)
    with col4:
        if fairness_available:
            if not bias_detected:
                st.success("✅ Fair")
            else:
                st.error("⚠️ Bias Detected")
        else:
            st.info("ℹ️ Fairness Data Unavailable")
    
    st.markdown("---")
    
    # ========================================================================
    # SECTION 6: MODEL METADATA
    # ========================================================================
    with st.expander("📊 View Model Metadata"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Information")
            if model_metrics:
                st.json({
                    "Train Size": model_metrics.get('train_size', 'N/A'),
                    "Test Size": model_metrics.get('test_size', 'N/A'),
                    "Dataset Size": model_metrics.get('dataset_size', 'N/A'),
                    "Accuracy": f"{model_metrics.get('accuracy', 0):.4f}"
                })
        
        with col2:
            st.subheader("Fairness Metrics")
            if fairness_data:
                st.json({
                    "Bias Detected": fairness_data.get('bias_detected', False),
                    "Overall Accuracy": f"{fairness_data.get('overall_accuracy', 0):.4f}",
                    "Demographic Parity": f"{fairness_data.get('demographic_parity', 0):.4f}",
                    "Timestamp": fairness_data.get('timestamp', 'N/A')
                })


if __name__ == "__main__":
    main()

