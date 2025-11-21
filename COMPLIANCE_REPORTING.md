# Compliance Reporting Documentation

## Overview

The compliance reporting system generates comprehensive PDF reports for Responsible AI governance. This document provides detailed information about the compliance reporting process, report structure, and customization options.

## Table of Contents

- [Purpose](#purpose)
- [Report Generation Process](#report-generation-process)
- [Report Structure](#report-structure)
- [Compliance Standards](#compliance-standards)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Purpose

The compliance reporting system serves multiple purposes:

1. **Regulatory Compliance**: Meet requirements for AI governance frameworks (EU AI Act, GDPR, ISO/IEC 23053)
2. **Stakeholder Communication**: Provide clear, professional reports for business leaders
3. **Audit Trail**: Document model performance, fairness, and explainability for auditors
4. **Transparency**: Demonstrate Responsible AI practices and model accountability

## Report Generation Process

### Automatic Generation

The compliance report is automatically generated when running the complete pipeline:

```bash
python run_pipeline.py
```

This executes:
1. ML pipeline (`ml_pipeline.py`) - generates all artifacts
2. Report generation (`make_compliance_report.py`) - creates PDF from artifacts

### Manual Generation

To generate a report from existing artifacts:

```bash
python make_compliance_report.py
```

**Prerequisites:**
- Artifacts must exist in `artifacts/` directory
- Minimum required files:
  - `model_metrics.json`
  - `fairness_report.json`

### Required Artifacts

The report generator loads the following artifacts:

| Artifact File | Description | Required |
|--------------|-------------|----------|
| `model_metrics.json` | Model performance metrics | ✅ Yes |
| `fairness_report.json` | Fairness analysis results | ✅ Yes |
| `drift_report.csv` | Drift detection results | ⚠️ Optional |
| `lime_meta.json` | LIME explanations | ⚠️ Optional |
| `shap_meta.json` | SHAP explanations | ⚠️ Optional |
| `shap_*.png` | SHAP visualizations | ⚠️ Optional |
| `logs/alert_logs.csv` | Human oversight logs | ⚠️ Optional |

**Note**: Missing optional artifacts will result in "N/A" or "Not available" messages in the report, but the report will still be generated.

## Report Structure

### 1. Executive Summary

**Purpose**: High-level overview for executives and stakeholders

**Contents**:
- Report generation timestamp
- Test set accuracy
- Full dataset accuracy
- Precision, Recall, F1 Score
- Data drift status
- Cross-type performance equity status
- Demographic parity metric
- Explainability availability (LIME/SHAP)

**Example Output**:
```
Report Date: 2024-01-15T10:30:00 UTC
Test Set Accuracy: 99.10%
Full Dataset Accuracy: 99.10%
Precision: 87.50%
Recall: 85.20%
F1 Score: 86.30%
Data Drift: 10.0% (1/10 features)
Cross-Type Performance Equity: No major bias ✅
Demographic Parity: 0.5%
Explainability: LIME + SHAP ✅
```

### 2. Model Performance

**Purpose**: Detailed model performance metrics

**Contents**:
- Test set accuracy
- Training and test sample sizes
- Dataset information and notes
- Top 5 feature importances
- Feature importance validation

**Key Metrics**:
- **Accuracy**: Overall prediction correctness
- **Feature Importance**: Which features contribute most to predictions
- **Sample Sizes**: Training/test split information

### 3. Governance Overview

**Purpose**: Demonstrate Responsible AI governance across MLOps lifecycle

**Contents**:
- Pre-development: Requirements and dataset balance
- Development: Model interpretability (LIME, SHAP)
- Deployment: Human-in-the-loop and audit logging
- Post-deployment: Drift monitoring and fairness tracking
- Model architecture details
- Feature engineering rationale
- Failure mode statistics

**Governance Principles Covered**:
- ✅ Transparency
- ✅ Fairness
- ✅ Accountability
- ✅ Robustness
- ✅ Privacy
- ✅ Human oversight

### 4. Data Drift Monitoring

**Purpose**: Monitor data distribution shifts over time

**Contents**:
- Number of monitored features
- Features with detected drift
- Drift detection summary table
- Statistical test results (p-values)

**Drift Detection Method**:
- **Test**: Kolmogorov-Smirnov (KS) test
- **Significance Level**: p < 0.05 indicates drift
- **Features Monitored**: All 10 model features

**Interpretation**:
- **No Drift (p ≥ 0.05)**: Data distribution stable
- **Drift Detected (p < 0.05)**: Distribution shift detected, model may need retraining

### 5. Cross-Type Performance Equity Assessment

**Purpose**: Ensure fair performance across machine types (L/M/H)

**Contents**:
- Overall model performance (full dataset)
- Bias detection status
- Demographic parity metric
- Performance metrics by machine type:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Actual Failure Rate
  - Predicted Failure Rate
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)
- FNR disparity analysis (if detected)
- Root cause analysis
- Recommendations

**Key Metrics**:

**Demographic Parity**: Difference in positive prediction rates across types
- **0.0**: Perfect parity (ideal)
- **< 0.05**: Good parity
- **> 0.10**: Significant disparity (requires attention)

**False Negative Rate (FNR)**: Critical metric for predictive maintenance
- **FNR = Missed Failures / Total Failures**
- **Lower is better**: Missed failures can cause equipment damage
- **Disparity Threshold**: FNR difference > 5 percentage points or ratio > 2.0x

**Bias Detection Criteria**:
- Predicted failure rate differs from actual by > 10%
- FPR/FNR difference > 15% within same type
- FNR disparity > 5 percentage points across types
- FNR ratio > 2.0x between types

### 6. Explainability (LIME)

**Purpose**: Local interpretability for individual predictions

**Contents**:
- Number of features analyzed
- Model classes
- Sample explanations (failure and non-failure cases)
- Top contributing features per sample
- Prediction probabilities

**LIME Explanation Format**:
- **Feature**: Feature name
- **Importance**: Contribution to prediction (positive = increases failure probability, negative = decreases)

**Example**:
```
Sample Type: FAILURE
• Tool_wear_Torque: +0.35
• Temp_diff: +0.22
• Power [W]: +0.18
• OSF_risk: +0.15
• Torque [Nm]: +0.10

Prediction: No Failure: 12.3%, Failure: 87.7%
```

### 7. Explainability (SHAP)

**Purpose**: Global and local feature importance using Shapley values

**Contents**:
- Number of features analyzed
- Global feature importance (mean |SHAP value|)
- SHAP visualizations:
  - Feature importance bar chart
  - Summary plot (all features)
- Individual sample explanations (if available)
- Feature contributions with SHAP values

**SHAP Explanation Format**:
- **Base Value**: Expected model output
- **SHAP Value**: Feature contribution (positive = increases prediction, negative = decreases)
- **Feature Value**: Actual feature value for the sample

**Visualizations**:
- **Feature Importance Chart**: Bar chart of mean |SHAP values|
- **Summary Plot**: Shows SHAP values for all samples, colored by feature value

### 8. Human Oversight Logs

**Purpose**: Track human interventions and decisions

**Contents**:
- Total logged decisions
- Recent intervention log entries (last 5)
- Intervention details (timestamp, action, reason)

**Log Format** (CSV):
```csv
timestamp,action,reason,model_prediction,human_decision
2024-01-15 10:30:00,OVERRIDE,High confidence false positive,FAILURE,NO_FAILURE
```

**Note**: If no logs exist, the report will indicate this is normal (no interventions occurred).

### 9. Compliance Notes

**Purpose**: Document compliance measures and policies

**Contents**:
- **Transparency**: Model explanations available
- **Human Oversight**: Critical predictions require approval
- **Data Retention**: 15-year retention policy
- **Bias Prevention**: Balanced dataset and continuous monitoring
- **Drift Monitoring**: Statistical tests applied
- **Model Updates**: Quarterly retraining or when drift exceeds threshold
- **Documentation**: Full model documentation available

## Compliance Standards

The compliance reporting system is designed to meet:

### EU AI Act (2024)
- **High-Risk AI Systems**: Predictive maintenance qualifies as high-risk
- **Requirements Met**:
  - Risk management system ✅
  - Data governance ✅
  - Technical documentation ✅
  - Record keeping ✅
  - Transparency ✅
  - Human oversight ✅
  - Accuracy and robustness ✅

### GDPR (General Data Protection Regulation)
- **Requirements Met**:
  - Data minimization ✅
  - Purpose limitation ✅
  - Transparency ✅
  - Accountability ✅
  - Right to explanation ✅

### ISO/IEC 23053 (2022)
- **Framework for AI Systems**: Requirements for AI system documentation
- **Requirements Met**:
  - System documentation ✅
  - Performance metrics ✅
  - Risk assessment ✅
  - Explainability ✅

### IEEE 7000 (2021)
- **Ethical Design**: Standards for ethical design of autonomous systems
- **Requirements Met**:
  - Ethical considerations ✅
  - Stakeholder engagement ✅
  - Transparency ✅
  - Accountability ✅

## Customization

### Adding Custom Sections

Edit `make_compliance_report.py` to add new sections:

```python
# Add custom section after existing sections
story.append(Paragraph("<b>8. Custom Section</b>", styles["Heading3"]))
story.append(Paragraph("Your custom content here", styles["BodyText"]))
story.append(Spacer(1, 12))
```

### Modifying Report Styling

**Change Colors**:
```python
# In TableStyle definitions
("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#your-color"))
```

**Change Fonts**:
```python
# Register custom font
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('CustomFont', 'path/to/font.ttf'))
```

**Change Page Layout**:
```python
# Modify SimpleDocTemplate parameters
doc = SimpleDocTemplate(
    str(REPORT_PATH),
    pagesize=A4,
    rightMargin=72,  # Adjust margins
    leftMargin=72,
    topMargin=72,
    bottomMargin=72
)
```

### Including Additional Metrics

**Load Custom JSON**:
```python
custom_metrics = safe_load_json(ARTIFACTS / "custom_metrics.json")
if custom_metrics:
    story.append(Paragraph(f"Custom Metric: {custom_metrics.get('value')}", styles["BodyText"]))
```

**Add Custom Tables**:
```python
custom_data = [
    ["Metric", "Value"],
    ["Custom 1", "100"],
    ["Custom 2", "200"]
]
custom_table = Table(custom_data)
custom_table.setStyle(TableStyle([...]))
story.append(custom_table)
```

### Modifying Executive Summary

Edit the `build_summary()` function in `make_compliance_report.py`:

```python
def build_summary(drift_df, fairness_data, lime_meta, shap_meta, model_metrics):
    # Add custom metrics
    summary = {
        # ... existing metrics ...
        "Custom Metric": calculate_custom_metric()
    }
    return summary
```

## Troubleshooting

### Report Generation Fails

**Issue**: `FileNotFoundError` when loading artifacts

**Solution**:
1. Ensure ML pipeline has been run: `python ml_pipeline.py`
2. Verify artifacts exist in `artifacts/` directory
3. Check file permissions

**Issue**: PDF generation fails with `ImportError`

**Solution**:
```bash
pip install reportlab
```

**Issue**: Images not appearing in PDF

**Solution**:
1. Verify SHAP images exist in `artifacts/` directory
2. Check image file paths are correct
3. Ensure images are valid PNG files

### Report Content Issues

**Issue**: "N/A" or "Not available" for all metrics

**Solution**:
1. Run ML pipeline first to generate artifacts
2. Check artifact files are valid JSON/CSV
3. Verify artifact file structure matches expected format

**Issue**: Missing sections in report

**Solution**:
1. Check that corresponding artifacts exist
2. Verify artifact file names match expected names
3. Review `make_compliance_report.py` for section conditions

### Performance Issues

**Issue**: Report generation is slow

**Solution**:
1. Reduce image sizes (SHAP plots)
2. Limit number of samples in explanations
3. Use smaller sample sizes for SHAP calculations

## Best Practices

1. **Generate Reports Regularly**: Create reports after each model retraining
2. **Version Control**: Keep historical reports for audit trail
3. **Review Before Distribution**: Always review reports before sharing with stakeholders
4. **Archive Artifacts**: Preserve artifacts used to generate reports
5. **Document Customizations**: Document any customizations made to report generation

## Additional Resources

- **ReportLab Documentation**: https://www.reportlab.com/docs/
- **EU AI Act**: https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- **GDPR**: https://gdpr.eu/
- **ISO/IEC 23053**: https://www.iso.org/standard/74438.html

---

**Last Updated**: 2024

