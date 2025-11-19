# Predictive Maintenance ML Pipeline

A complete machine learning pipeline for predictive maintenance with Responsible AI compliance reporting.

## Features

- **Random Forest Model**: Predicts machine failures using 14 engineered features
- **Model Explainability**: LIME and SHAP explanations for interpretable predictions
- **Drift Detection**: Statistical monitoring of data distribution shifts
- **Fairness Analysis**: Performance evaluation across machine types (L/M/H)
- **Compliance Reporting**: Automated PDF generation for Responsible AI governance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:

```bash
python run_pipeline.py
```

This executes:
1. Data preprocessing and feature engineering
2. Model training and evaluation
3. Explainability analysis (LIME & SHAP)
4. Drift detection
5. Fairness assessment
6. Compliance report generation

### Individual Components

```bash
# Train model and generate artifacts
python ml_pipeline.py

# Generate compliance report from artifacts
python make_compliance_report.py
```

## Dataset

Uses the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/inIT-OWL/predictive-maintenance-dataset-ai4i-2020) with:
- 10,000 samples
- 14 features (temperature, rotational speed, torque, tool wear, etc.)
- Binary classification target (machine failure)

## Output

### Artifacts (`artifacts/`)
- `rf_model.pkl` - Trained Random Forest model
- `model_metrics.json` - Performance metrics
- `lime_meta.json` - LIME explanations
- `shap_meta.json` - SHAP explanations and visualizations
- `drift_report.csv` - Drift detection results
- `fairness_report.json` - Fairness analysis by machine type

### Reports
- `compliance_report.pdf` - Comprehensive Responsible AI compliance report

## Model Architecture

- **Algorithm**: Random Forest Classifier (100 trees, max depth 10)
- **Features**: 10 engineered features including domain-specific metrics
- **Performance**: ~99% accuracy with balanced class weights

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- scipy, lime, shap
- matplotlib, reportlab

See `requirements.txt` for complete dependencies.

## License

MIT License
