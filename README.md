# Predictive Maintenance ML Pipeline

A comprehensive machine learning pipeline for predictive maintenance with Responsible AI compliance reporting, explainability, drift detection, and fairness analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Compliance Reporting](#compliance-reporting)
- [Output Artifacts](#output-artifacts)
- [Testing](#testing)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for predicting machine failures in industrial equipment. The system includes:

- **ML Model Training**: Random Forest classifier with engineered features
- **Explainability**: LIME and SHAP explanations for model interpretability
- **Drift Detection**: Statistical monitoring of data distribution shifts
- **Fairness Analysis**: Performance evaluation across machine types (L/M/H)
- **Compliance Reporting**: Automated PDF generation for Responsible AI governance

The pipeline is designed with Responsible AI principles in mind, ensuring transparency, fairness, and compliance with regulatory requirements.

## ✨ Features

### Core ML Pipeline
- **Random Forest Model**: Predicts machine failures using 10 engineered features
- **Feature Engineering**: Domain-specific features based on failure mode physics
  - Temperature difference (Heat Dissipation Failure detection)
  - Power calculation (Power Failure detection)
  - Tool wear × Torque (Overstrain Failure detection)
  - OSF Risk Score (normalized stress level)
- **Model Performance**: ~99% accuracy with balanced class weights

### Explainability
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local explanations for individual predictions
- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
  - Summary plots showing feature contributions
  - Feature importance visualizations
  - Individual sample explanations

### Monitoring & Analysis
- **Drift Detection**: Kolmogorov-Smirnov statistical tests to detect data distribution shifts
- **Fairness Analysis**: Comprehensive performance evaluation across machine types
  - Accuracy, Precision, Recall, F1 Score by type
  - False Positive Rate (FPR) and False Negative Rate (FNR) analysis
  - Demographic parity metrics
  - Cross-type performance equity assessment

### Compliance & Governance
- **Automated Compliance Reports**: Professional PDF reports for stakeholders and auditors
- **Human Oversight Logging**: Track human interventions and decisions
- **Model Documentation**: Complete model documentation with feature importance
- **Audit Trail**: All artifacts saved for regulatory compliance

### Monitoring Dashboard
- **Real-time Monitoring**: Interactive Streamlit dashboard for model health
- **Performance Metrics**: Visual display of accuracy, precision, recall, F1 score
- **Drift Visualization**: Interactive charts showing drift detection results
- **Fairness Dashboard**: Visual comparison of performance across machine types
- **Feature Importance**: Interactive charts for model and SHAP feature importance
- **System Health**: Real-time status of model, explainability, drift, and fairness

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "predictive manteiance"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Dataset
Ensure the dataset file is located at:
```
dataset/ai4i2020.csv
```

The dataset should contain the following required columns:
- `Type` (L/M/H)
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Machine failure`
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (failure mode indicators)

## 🏃 Quick Start

### Run Complete Pipeline
Execute the full pipeline (training + compliance report):
```bash
python run_pipeline.py
```

This will:
1. Load and preprocess the dataset
2. Train the Random Forest model
3. Generate LIME and SHAP explanations
4. Perform drift detection
5. Analyze fairness across machine types
6. Generate compliance report PDF

### Run Individual Components

**Train Model Only:**
```bash
python ml_pipeline.py
```

**Generate Compliance Report Only:**
```bash
python make_compliance_report.py
```

**Launch Monitoring Dashboard:**
```bash
# Option 1: Default port (8501)
streamlit run dashboard.py

# Option 2: Alternative port (if 8501 is busy)
streamlit run dashboard.py --server.port 8502

# Option 3: Use the batch file (Windows)
start_dashboard.bat
```

The dashboard provides real-time monitoring of:
- Model performance metrics
- Data drift detection
- Fairness analysis by machine type
- Feature importance visualizations
- System health status
- Graphviz decision tree visualization

**Access the dashboard at:** `http://localhost:8501` (or the port you specified)

**Troubleshooting:** If the dashboard won't start, see `QUICK_START_DASHBOARD.md` for solutions.

## 📁 Project Structure

```
predictive-manteiance/
│
├── ml_pipeline.py              # Main ML pipeline script
├── make_compliance_report.py   # Compliance report generator
├── run_pipeline.py             # Entry point for complete pipeline
├── dashboard.py                # Streamlit monitoring dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── dataset/
│   └── ai4i2020.csv            # Training dataset
│
├── artifacts/                  # Generated outputs
│   ├── rf_model.pkl            # Trained model
│   ├── model_metrics.json     # Performance metrics
│   ├── lime_meta.json          # LIME explanations
│   ├── shap_meta.json          # SHAP explanations
│   ├── drift_report.csv        # Drift detection results
│   ├── fairness_report.json    # Fairness analysis
│   └── *.png                   # SHAP visualizations
│
├── logs/                       # Human oversight logs
│   └── alert_logs.csv          # Intervention logs
│
├── tests/                      # Unit tests
│   ├── test_ml_pipeline.py     # Tests for ML pipeline
│   └── test_compliance_report.py # Tests for report generation
│
└── compliance_report.pdf       # Generated compliance report
```

## 📖 Usage

### Basic Usage

**1. Run the complete pipeline:**
```bash
python run_pipeline.py
```

**2. Check outputs:**
- Model artifacts: `artifacts/`
- Compliance report: `compliance_report.pdf`

### Advanced Usage

#### Custom Model Training
Modify `ml_pipeline.py` to adjust:
- Model hyperparameters (n_estimators, max_depth)
- Feature engineering
- Sample weights
- Train/test split ratio

#### Custom Compliance Report
Modify `make_compliance_report.py` to:
- Add custom sections
- Change report styling
- Include additional metrics

## 📊 Dataset

### AI4I 2020 Predictive Maintenance Dataset

**Source**: [Kaggle - AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/inIT-OWL/predictive-maintenance-dataset-ai4i-2020)

**Characteristics:**
- **Size**: 10,000 samples
- **Features**: 14 original features + 4 engineered features
- **Target**: Binary classification (Machine failure: 0/1)
- **Machine Types**: L (Low quality), M (Medium), H (High quality)
- **Failure Modes**: TWF, HDF, PWF, OSF, RNF

**Expected Distribution:**
- Type L: ~50% of samples
- Type M: ~30% of samples
- Type H: ~20% of samples

**Failure Rate**: ~3.4% overall (imbalanced classification problem)

## 🏗️ Model Architecture

### Algorithm
- **Type**: Random Forest Classifier
- **Trees**: 100 estimators
- **Max Depth**: 10 (prevents overfitting)
- **Class Weight**: Balanced (handles class imbalance)
- **Random State**: 42 (reproducibility)

### Features (10 total)
1. `Type_encoded` - Machine quality type (0=L, 1=M, 2=H)
2. `Air temperature [K]` - Ambient temperature
3. `Process temperature [K]` - Operating temperature
4. `Temp_diff` - Temperature difference (engineered)
5. `Rotational speed [rpm]` - Machine rotation speed
6. `Torque [Nm]` - Rotational force
7. `Tool wear [min]` - Cumulative tool usage time
8. `Power [W]` - Calculated power (engineered)
9. `Tool_wear_Torque` - Combined stress metric (engineered)
10. `OSF_risk` - Overstrain failure risk score (engineered)

### Feature Engineering Details

**Temperature Difference:**
```
Temp_diff = Process_temp - Air_temp
```
- Critical for detecting Heat Dissipation Failure (HDF)
- HDF occurs when Temp_diff < 8.6K AND Rotational_speed < 1380 rpm

**Power Calculation:**
```
Power [W] = Torque [Nm] × Rotational_speed [rpm] × (2π / 60)
```
- Critical for detecting Power Failure (PWF)
- PWF occurs when Power < 3500W OR Power > 9000W

**Tool Wear × Torque:**
```
Tool_wear_Torque = Tool_wear [min] × Torque [Nm]
```
- Critical for detecting Overstrain Failure (OSF)
- OSF thresholds: L=11,000, M=12,000, H=13,000 min·Nm

**OSF Risk Score:**
```
OSF_risk = Tool_wear_Torque / OSF_threshold
```
- Normalized stress level (0.0-1.0 = safe, >1.0 = at risk)

### Physics-Based Feature Explanations

The engineered features in this model are grounded in fundamental mechanical and thermal physics principles that govern industrial machinery behavior:

**Temperature Difference (Heat Transfer Physics):**
The temperature difference (`Temp_diff = Process_temp - Air_temp`) represents the thermal gradient driving heat dissipation from the machine to its environment. According to Newton's law of cooling, the rate of heat transfer is proportional to this temperature difference. When `Temp_diff < 8.6K` combined with low rotational speed, the machine cannot effectively dissipate heat through forced convection (airflow), leading to Heat Dissipation Failure (HDF). This follows the heat transfer equation: `Q = h·A·ΔT`, where heat flux (Q) depends on the temperature difference (ΔT), surface area (A), and heat transfer coefficient (h), which is reduced at low rotational speeds.

**Power Calculation (Rotational Dynamics):**
The power feature (`Power = Torque × Rotational_speed × 2π/60`) derives from fundamental rotational mechanics. Power in rotating systems is the product of torque (τ) and angular velocity (ω): `P = τ·ω`. The conversion factor `2π/60` transforms rpm to rad/s. Power failures occur when the system operates outside the safe operating window (3500W < P < 9000W), indicating either insufficient power delivery (underpowered operation) or excessive power demand (overload conditions), both of which violate the machine's design specifications and lead to Power Failure (PWF).

**Tool Wear × Torque (Material Fatigue Physics):**
The `Tool_wear_Torque` metric combines cumulative damage (tool wear) with applied stress (torque), following the principles of fatigue failure and cumulative damage theory (Miner's rule). As tools wear, their material properties degrade, reducing the maximum stress they can withstand. The product `Tool_wear × Torque` represents accumulated stress cycles, where each operating minute at a given torque level contributes to material fatigue. When this accumulated stress exceeds the material's endurance limit (thresholds: L=11,000, M=12,000, H=13,000 min·Nm), the material undergoes Overstrain Failure (OSF) due to progressive microcrack propagation and eventual fracture.

**OSF Risk Score (Normalized Stress Analysis):**
The OSF risk score normalizes the accumulated stress relative to the material's failure threshold, following the stress-life (S-N) curve relationship. Values < 1.0 indicate operation within the safe fatigue limit, while values > 1.0 indicate operation beyond the material's endurance limit, where failure becomes probable. This follows the relationship: `σ_applied / σ_endurance = risk_factor`, where exceeding unity indicates imminent failure.

### Performance Metrics
- **Accuracy**: ~99.10% (test set)
- **Precision**: ~85-90%
- **Recall**: ~85-90%
- **F1 Score**: ~85-90%

## 📄 Compliance Reporting

### Overview

The compliance reporting system generates comprehensive PDF reports for Responsible AI governance. These reports are designed for:
- **Stakeholders**: Business leaders and decision-makers
- **Auditors**: Regulatory compliance reviewers
- **Technical Teams**: ML engineers and data scientists

### Report Sections

The generated `compliance_report.pdf` includes:

1. **Executive Summary**
   - Model performance metrics
   - Drift detection status
   - Fairness assessment
   - Explainability availability

2. **Model Performance**
   - Test set accuracy
   - Training/test sample sizes
   - Feature importance rankings
   - Performance notes

3. **Governance Overview**
   - MLOps lifecycle compliance
   - Model type and features
   - Domain knowledge integration
   - Failure mode statistics

4. **Data Drift Monitoring**
   - Feature-level drift detection
   - Statistical test results (p-values)
   - Distribution comparisons

5. **Cross-Type Performance Equity Assessment**
   - Performance metrics by machine type (L/M/H)
   - Bias detection results
   - Demographic parity metrics
   - False Negative Rate (FNR) analysis
   - Root cause analysis for disparities

6. **Explainability (LIME)**
   - Feature-level explanations
   - Sample predictions with reasoning
   - Top contributing features

7. **Explainability (SHAP)**
   - Global feature importance
   - Summary plots (visualizations)
   - Individual sample explanations
   - Feature contribution analysis

8. **Human Oversight Logs**
   - Intervention tracking
   - Decision audit trail

9. **Compliance Notes**
   - Transparency measures
   - Data retention policies
   - Bias prevention strategies
   - Model update procedures

### Generating Reports

**Automatic (with pipeline):**
```bash
python run_pipeline.py
```

**Manual:**
```bash
python make_compliance_report.py
```

**Requirements:**
- Artifacts must exist in `artifacts/` directory
- At minimum: `model_metrics.json` and `fairness_report.json`

### Report Customization

To customize reports, edit `make_compliance_report.py`:

**Add Custom Sections:**
```python
story.append(Paragraph("<b>Custom Section</b>", styles["Heading3"]))
story.append(Paragraph("Your content here", styles["BodyText"]))
```

**Modify Styling:**
```python
# Change colors, fonts, etc. in TableStyle definitions
```

**Include Additional Metrics:**
```python
# Load custom JSON/CSV files
custom_data = safe_load_json(ARTIFACTS / "custom_metrics.json")
```

### Compliance Standards

The reports are designed to meet:
- **EU AI Act** requirements for high-risk AI systems
- **GDPR** data protection standards
- **ISO/IEC 23053** framework for AI systems
- **IEEE 7000** standards for ethical design

### Key Compliance Features

- ✅ **Transparency**: Full model documentation and explanations
- ✅ **Fairness**: Cross-type performance equity analysis
- ✅ **Accountability**: Human oversight logging
- ✅ **Robustness**: Drift detection and monitoring
- ✅ **Privacy**: Data retention policies documented
- ✅ **Auditability**: Complete artifact preservation

## 📦 Output Artifacts

### Model Files
- **`rf_model.pkl`**: Trained Random Forest model (pickle format)
- **`rf_model_backup_*.pkl`**: Timestamped backups of previous models

### Metrics & Reports
- **`model_metrics.json`**: Model performance metrics
  ```json
  {
    "accuracy": 0.9910,
    "train_size": 8000,
    "test_size": 2000,
    "feature_importance": {...}
  }
  ```

- **`fairness_report.json`**: Fairness analysis by machine type
  ```json
  {
    "bias_detected": false,
    "overall_accuracy": 0.9910,
    "by_type": {
      "L": {...},
      "M": {...},
      "H": {...}
    }
  }
  ```

- **`drift_report.csv`**: Drift detection results per feature
- **`lime_meta.json`**: LIME explanations for sample predictions
- **`shap_meta.json`**: SHAP global importance and sample explanations

### Visualizations
- **`shap_summary_plot.png`**: SHAP summary plot (all features)
- **`shap_summary_plot_custom.png`**: Custom SHAP plot (guaranteed all features)
- **`shap_feature_importance.png`**: Feature importance bar chart

### Human Oversight Logs
- **`logs/alert_logs.csv`**: Human intervention and decision audit trail

**Mock Log Example:**
```csv
timestamp,action,reason,model_prediction,human_decision
2024-01-15 10:30:00,OVERRIDE,High confidence false positive detected,FAILURE,NO_FAILURE
2024-01-15 11:15:23,APPROVED,Model prediction confirmed by maintenance team,FAILURE,FAILURE
2024-01-15 14:42:10,OVERRIDE,Contextual information: recent maintenance performed,FAILURE,NO_FAILURE
2024-01-16 09:20:45,APPROVED,SHAP explanation shows high Tool_wear_Torque risk,NO_FAILURE,FAILURE
2024-01-16 15:33:12,APPROVED,Standard prediction within acceptable confidence range,NO_FAILURE,NO_FAILURE
```

The log tracks all human interventions where operators override or approve model predictions, providing a complete audit trail for regulatory compliance and model improvement.

### Model Visualization (Graphviz)
- **`model_decision_tree.dot`**: Graphviz DOT format file for visualizing Random Forest decision trees
- **`model_decision_tree.png`**: Rendered decision tree visualization (if Graphviz is installed)

**Graphviz Integration:**
The pipeline can generate Graphviz DOT format files to visualize individual decision trees from the Random Forest ensemble. Graphviz uses the DOT language to create graph visualizations, allowing inspection of the decision paths and feature splits that the model uses for predictions. To generate visualizations:

1. **Install Graphviz**: 
   ```bash
   # Windows (using Chocolatey)
   choco install graphviz
   
   # macOS (using Homebrew)
   brew install graphviz
   
   # Linux (Ubuntu/Debian)
   sudo apt-get install graphviz
   ```

2. **Python Package**:
   ```bash
   pip install graphviz
   ```

3. **Generate Visualization**:
   The pipeline can export individual trees in DOT format, which can be rendered using:
   ```bash
   dot -Tpng model_decision_tree.dot -o model_decision_tree.png
   ```

Graphviz visualizations help explain model decisions by showing the exact feature thresholds and decision paths that lead to failure predictions, enhancing model interpretability and compliance with explainability requirements.

### Compliance Report
- **`compliance_report.pdf`**: Complete Responsible AI compliance report

## 🧪 Testing

### Running Tests

**Run all tests:**
```bash
pytest tests/
```

**Run specific test file:**
```bash
pytest tests/test_ml_pipeline.py
pytest tests/test_compliance_report.py
```

**Run with coverage:**
```bash
pytest tests/ --cov=ml_pipeline --cov=make_compliance_report --cov-report=html
```

### Test Structure

Tests are organized by module:
- **`test_ml_pipeline.py`**: Tests for ML pipeline functions
  - Data loading and preprocessing
  - Model training
  - Explainability generation
  - Drift detection
  - Fairness analysis

- **`test_compliance_report.py`**: Tests for report generation
  - JSON/CSV loading
  - PDF generation
  - Report content validation

### Writing New Tests

Example test structure:
```python
import pytest
from ml_pipeline import load_and_preprocess_data

def test_load_data():
    X, y, df, feature_cols, le = load_and_preprocess_data()
    assert not X.empty
    assert len(y) > 0
    assert len(feature_cols) == 10
```

## 📋 Requirements

### Python Packages

See `requirements.txt` for complete list:
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.23.0` - Numerical computing
- `scikit-learn>=1.2.0` - Machine learning
- `scipy>=1.9.0` - Statistical tests
- `lime>=0.2.0` - LIME explainability (optional)
- `shap>=0.42.0` - SHAP explainability (optional)
- `matplotlib>=3.5.0` - Visualizations
- `reportlab>=3.6.0` - PDF generation
- `pytest>=7.0.0` - Testing framework (optional, for tests)

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 500MB for dataset and artifacts
- **OS**: Windows, Linux, or macOS

## 🔧 Troubleshooting

### Common Issues

**1. Dataset Not Found**
```
FileNotFoundError: Dataset not found at dataset/ai4i2020.csv
```
**Solution**: Ensure the dataset file exists in the `dataset/` directory.

**2. Missing Dependencies**
```
ImportError: No module named 'lime'
```
**Solution**: Install missing packages:
```bash
pip install -r requirements.txt
```

**3. LIME/SHAP Not Available**
The pipeline will continue without LIME/SHAP if not installed. To enable:
```bash
pip install lime shap
```

**4. Memory Issues**
If you encounter memory errors with large datasets:
- Reduce dataset size for testing
- Use smaller sample sizes in SHAP calculations
- Process data in batches

**5. PDF Generation Fails**
If report generation fails:
- Check that artifacts exist in `artifacts/` directory
- Verify `reportlab` is installed: `pip install reportlab`
- Check file permissions for writing PDF

**6. Model Training Fails**
If training fails:
- Verify dataset has all required columns
- Check for missing values in critical features
- Ensure minimum dataset size (20 samples)

### Getting Help

1. Check the error message and logs
2. Verify all dependencies are installed
3. Ensure dataset is in correct format
4. Review the troubleshooting section above
5. Check GitHub issues (if applicable)

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Code Style
- Follow PEP 8 Python style guide
- Use meaningful variable names
- Add docstrings to functions
- Include type hints where appropriate

### Testing
- Add unit tests for new functions
- Ensure all tests pass: `pytest tests/`
- Maintain or improve test coverage

## 📚 Documentation

### Available Documentation

This project includes comprehensive documentation for Responsible AI practices in predictive maintenance:

1. **README.md** (This file)
   - Complete project overview and usage guide
   - Installation and setup instructions
   - Model architecture and feature engineering details
   - Testing and troubleshooting guides

2. **COMPLIANCE_REPORTING.md**
   - Detailed guide to compliance reporting system
   - Report structure and sections explained
   - Compliance standards (EU AI Act, GDPR, ISO/IEC 23053)
   - Customization and troubleshooting

3. **DASHBOARD_GUIDE.md**
   - Monitoring dashboard usage guide
   - Feature descriptions and usage tips
   - Customization instructions
   - Troubleshooting guide

4. **Responsible AI for Predictive Maintenance PDF**
   - Comprehensive guide on Responsible AI principles
   - Best practices for predictive maintenance systems
   - Compliance frameworks and regulatory requirements
   - Implementation guidelines
   
   **Location**: `Responsible AI for Predictive Maintenance_ A Compl... .pdf`

5. **Generated Compliance Reports**
   - `compliance_report.pdf` - Automated Responsible AI compliance report
   - Generated after running the ML pipeline
   - Includes model performance, fairness analysis, drift detection, and explainability

### Documentation Structure

```
predictive-manteiance/
├── README.md                              # Main project documentation
├── COMPLIANCE_REPORTING.md                # Compliance reporting guide
├── DASHBOARD_GUIDE.md                     # Monitoring dashboard guide
├── Responsible AI for Predictive Maintenance_ A Compl... .pdf  # Responsible AI guide
├── compliance_report.pdf                  # Generated compliance report
└── Predictive_Maintenance_System_Guide.docx  # System guide (Word format)
```

### Quick Links

- **Getting Started**: See [Installation](#-installation) and [Quick Start](#-quick-start)
- **Compliance**: See [Compliance Reporting](#-compliance-reporting) section
- **Monitoring**: See [Monitoring Dashboard](#monitoring-dashboard) section
- **Testing**: See [Testing](#-testing) section

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Dataset**: AI4I 2020 Predictive Maintenance Dataset (Kaggle)
- **Libraries**: scikit-learn, LIME, SHAP, ReportLab
- **Inspiration**: Responsible AI best practices and MLOps principles

## 📞 Contact & Support

For questions, issues, or contributions:
- **GitHub Issues**: Open an issue on the repository
- **Email**: tshapedconsultant@gmail.com

---

**Built with ❤️ for Responsible AI**

*Last updated: 2024*
