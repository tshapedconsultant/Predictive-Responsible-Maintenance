# Monitoring Dashboard Guide

## Overview

The monitoring dashboard provides real-time visualization and monitoring of the Predictive Maintenance ML Pipeline. It displays model performance metrics, drift detection results, fairness analysis, and system health status.

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate Artifacts (if not already done)

The dashboard reads from the `artifacts/` directory. Ensure you've run the ML pipeline first:

```bash
python run_pipeline.py
```

This generates:
- `model_metrics.json` - Model performance metrics
- `fairness_report.json` - Fairness analysis results
- `drift_report.csv` - Drift detection results
- `lime_meta.json` - LIME explanations
- `shap_meta.json` - SHAP explanations

### 3. Launch Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Dashboard Features

### 1. Model Performance Overview
- **Test Accuracy**: Overall model accuracy on test set
- **Precision**: Percentage of positive predictions that are correct
- **Recall**: Percentage of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall

### 2. Data Drift Monitoring
- **Drift Summary**: Number of features with detected drift
- **Drift Rate**: Percentage of features showing drift
- **Visualization**: Interactive bar chart showing p-values for each feature
- **Status Indicators**: Color-coded drift status (green/yellow/red)

### 3. Fairness Analysis
- **Bias Detection**: Overall bias status
- **Demographic Parity**: Difference in prediction rates across types
- **Performance by Type**: Metrics comparison for L/M/H machine types
- **FNR Analysis**: Critical false negative rate analysis
- **Visualizations**: Bar charts comparing accuracy and FNR across types

### 4. Feature Importance
- **Model Feature Importance**: Random Forest feature importance scores
- **SHAP Global Importance**: SHAP-based feature importance (if available)
- **Interactive Charts**: Horizontal bar charts sorted by importance

### 5. System Health Status
- **Model Availability**: Check if model file exists
- **Explainability Status**: LIME and SHAP availability
- **Drift Monitoring**: Current drift detection status
- **Fairness Status**: Bias detection status

### 6. Model Metadata
- **Training Information**: Train/test sizes, dataset size, accuracy
- **Fairness Metrics**: Bias status, demographic parity, timestamp

## Dashboard Sections

### Sidebar
- **Auto-refresh**: Toggle automatic refresh every 30 seconds
- **Data Sources**: Status of artifact files (✅ available, ❌ missing)
- **Last Updated**: Timestamp of last dashboard update

### Main Content
- **Performance Metrics**: Key metrics displayed as cards
- **Interactive Charts**: Plotly charts for visual analysis
- **Data Tables**: Expandable tables with detailed information
- **Status Indicators**: Color-coded health indicators

## Usage Tips

### Auto-Refresh
Enable auto-refresh in the sidebar to keep the dashboard updated automatically. This is useful for monitoring during model training or when artifacts are being updated.

### Data Availability
The dashboard gracefully handles missing artifacts:
- Missing files show "N/A" or "Not available" messages
- The dashboard remains functional with partial data
- Status indicators show which artifacts are available

### Interactivity
- **Hover**: Hover over charts to see detailed values
- **Zoom**: Click and drag to zoom into chart areas
- **Pan**: Use toolbar to pan across charts
- **Export**: Use Plotly toolbar to export charts as images

### Performance
- Data is cached using Streamlit's `@st.cache_data` decorator
- Dashboard loads quickly even with large datasets
- Charts render efficiently using Plotly

## Troubleshooting

### Dashboard Shows "No Artifacts Found"
**Solution**: Run the ML pipeline first:
```bash
python run_pipeline.py
```

### Charts Not Displaying
**Solution**: 
1. Check that Plotly is installed: `pip install plotly`
2. Verify artifact files are valid JSON/CSV
3. Check browser console for JavaScript errors

### Dashboard Not Loading
**Solution**:
1. Verify Streamlit is installed: `pip install streamlit`
2. Check Python version (3.8+ required)
3. Review error messages in terminal

### Port Already in Use
**Solution**: Use a different port:
```bash
streamlit run dashboard.py --server.port 8502
```

## Customization

### Modify Dashboard Layout
Edit `dashboard.py` to:
- Add new sections
- Change chart types
- Modify color schemes
- Add custom metrics

### Add New Visualizations
Use Plotly to create custom charts:
```python
import plotly.express as px
fig = px.bar(data, x='x', y='y')
st.plotly_chart(fig, use_container_width=True)
```

### Customize Styling
Modify the CSS in the `st.markdown()` section at the top of `dashboard.py`:
```python
st.markdown("""
    <style>
    .custom-class {
        /* Your custom styles */
    }
    </style>
""", unsafe_allow_html=True)
```

## Best Practices

1. **Run Pipeline First**: Always generate artifacts before viewing dashboard
2. **Regular Updates**: Refresh dashboard after model retraining
3. **Monitor Drift**: Check drift section regularly for data quality issues
4. **Review Fairness**: Monitor fairness metrics to ensure equitable performance
5. **Export Reports**: Use dashboard insights to supplement compliance reports

## Integration with CI/CD

The dashboard can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Launch Dashboard
  run: |
    streamlit run dashboard.py --server.headless true
```

For production deployments, consider:
- Using Streamlit Cloud
- Docker containerization
- Reverse proxy setup
- Authentication/authorization

## Next Steps

- Add prediction logging to track model predictions over time
- Implement alerting for drift or performance degradation
- Create historical performance tracking
- Add model comparison capabilities
- Integrate with model registry

---

**Last Updated**: 2024

