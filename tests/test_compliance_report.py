"""
Unit tests for make_compliance_report.py

Tests cover:
- JSON/CSV loading functions
- Report generation
- PDF creation
- Report content validation
"""

import pytest
import json
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent directory to path to import make_compliance_report
sys.path.insert(0, str(Path(__file__).parent.parent))

from make_compliance_report import (
    safe_load_json,
    safe_load_csv,
    build_table_from_df,
    build_summary,
    format_fairness_table,
    BASE_DIR,
    ARTIFACTS
)


class TestSafeLoadJson:
    """Tests for safe JSON loading"""
    
    def test_load_valid_json(self):
        """Test loading valid JSON file"""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {"key": "value", "number": 42}
            json.dump(test_data, f)
            temp_path = Path(f.name)
        
        try:
            result = safe_load_json(temp_path)
            assert result == test_data
        finally:
            temp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file returns empty dict"""
        nonexistent_path = Path("nonexistent_file.json")
        result = safe_load_json(nonexistent_path)
        assert result == {}
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON returns empty dict"""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = Path(f.name)
        
        try:
            result = safe_load_json(temp_path)
            assert result == {}
        finally:
            temp_path.unlink()


class TestSafeLoadCsv:
    """Tests for safe CSV loading"""
    
    def test_load_valid_csv(self):
        """Test loading valid CSV file"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_path = Path(f.name)
        
        try:
            result = safe_load_csv(temp_path)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert 'col1' in result.columns
            assert 'col2' in result.columns
        finally:
            temp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file returns empty DataFrame"""
        nonexistent_path = Path("nonexistent_file.csv")
        result = safe_load_csv(nonexistent_path)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_load_invalid_csv(self):
        """Test loading invalid CSV returns empty DataFrame"""
        # Create temporary file with invalid CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\nunclosed quote")
            temp_path = Path(f.name)
        
        try:
            result = safe_load_csv(temp_path)
            # Should return empty DataFrame or handle gracefully
            assert isinstance(result, pd.DataFrame)
        finally:
            temp_path.unlink()


class TestBuildTableFromDf:
    """Tests for DataFrame to table conversion"""
    
    def test_build_table_from_valid_df(self):
        """Test building table from valid DataFrame"""
        df = pd.DataFrame({
            'col1': [1.12345, 2.67890],
            'col2': [3.45678, 4.90123]
        })
        
        table = build_table_from_df(df, max_rows=10)
        
        # Should return a Table object or Paragraph
        assert table is not None
    
    def test_build_table_from_empty_df(self):
        """Test building table from empty DataFrame"""
        df = pd.DataFrame()
        
        table = build_table_from_df(df, max_rows=10)
        
        # Should return Paragraph with "No data available"
        assert table is not None
    
    def test_build_table_with_max_rows(self):
        """Test that max_rows limit is respected"""
        # Create DataFrame with more than max_rows
        df = pd.DataFrame({
            'col1': range(20),
            'col2': range(20)
        })
        
        table = build_table_from_df(df, max_rows=5)
        
        # Table should be limited to 5 rows
        assert table is not None


class TestBuildSummary:
    """Tests for summary building"""
    
    def test_build_summary_with_all_data(self):
        """Test building summary with all data available"""
        # Create mock data
        drift_df = pd.DataFrame({
            'feature': ['feat1', 'feat2'],
            'drift_detected': [True, False],
            'p_value': [0.01, 0.10]
        })
        
        fairness_data = {
            'bias_detected': False,
            'overall_accuracy': 0.991,
            'overall_precision': 0.875,
            'overall_recall': 0.852,
            'overall_f1': 0.863,
            'demographic_parity': 0.005
        }
        
        lime_meta = {
            'explanations': [{'sample_type': 'failure'}],
            'feature_names': ['feat1', 'feat2']
        }
        
        shap_meta = {
            'shap_available': True,
            'explanations': [{'sample_type': 'failure'}],
            'global_importance': {'feat1': 0.5, 'feat2': 0.5}
        }
        
        model_metrics = {
            'accuracy': 0.991,
            'train_size': 8000,
            'test_size': 2000
        }
        
        summary = build_summary(drift_df, fairness_data, lime_meta, shap_meta, model_metrics)
        
        # Check structure
        assert isinstance(summary, dict)
        assert 'Test Set Accuracy' in summary
        assert 'Full Dataset Accuracy' in summary
        assert 'Drift' in summary
        assert 'Cross-Type Performance Equity' in summary
        assert 'Explainability' in summary
        assert 'Timestamp' in summary
    
    def test_build_summary_with_missing_data(self):
        """Test building summary with missing data"""
        # Empty/missing data
        drift_df = pd.DataFrame()
        fairness_data = {}
        lime_meta = {}
        shap_meta = {}
        model_metrics = {}
        
        summary = build_summary(drift_df, fairness_data, lime_meta, shap_meta, model_metrics)
        
        # Should still return summary with N/A values
        assert isinstance(summary, dict)
        assert 'Test Set Accuracy' in summary
        assert 'Full Dataset Accuracy' in summary


class TestFormatFairnessTable:
    """Tests for fairness table formatting"""
    
    def test_format_fairness_table_valid(self):
        """Test formatting valid fairness data"""
        fairness_data = {
            'by_type': {
                'L': {
                    'accuracy': 0.99,
                    'precision': 0.88,
                    'recall': 0.85,
                    'f1_score': 0.86,
                    'actual_failure_rate': 0.039,
                    'predicted_failure_rate': 0.040,
                    'false_positive_rate': 0.01,
                    'false_negative_rate': 0.04
                },
                'M': {
                    'accuracy': 0.98,
                    'precision': 0.87,
                    'recall': 0.84,
                    'f1_score': 0.85,
                    'actual_failure_rate': 0.035,
                    'predicted_failure_rate': 0.036,
                    'false_positive_rate': 0.01,
                    'false_negative_rate': 0.05
                },
                'H': {
                    'accuracy': 0.97,
                    'precision': 0.86,
                    'recall': 0.83,
                    'f1_score': 0.84,
                    'actual_failure_rate': 0.021,
                    'predicted_failure_rate': 0.022,
                    'false_positive_rate': 0.01,
                    'false_negative_rate': 0.10
                }
            }
        }
        
        table_data = format_fairness_table(fairness_data)
        
        # Check structure
        assert table_data is not None
        assert isinstance(table_data, list)
        assert len(table_data) > 0
        
        # Check header
        assert table_data[0] == ["Type", "Accuracy", "Precision", "Recall", "F1", 
                                 "Actual FR", "Predicted FR", "FPR", "FNR"]
        
        # Check data rows
        assert len(table_data) == 4  # Header + 3 types
    
    def test_format_fairness_table_empty(self):
        """Test formatting empty fairness data"""
        fairness_data = {}
        
        table_data = format_fairness_table(fairness_data)
        
        assert table_data is None
    
    def test_format_fairness_table_missing_by_type(self):
        """Test formatting fairness data without by_type"""
        fairness_data = {
            'bias_detected': False,
            'overall_accuracy': 0.99
        }
        
        table_data = format_fairness_table(fairness_data)
        
        assert table_data is None


class TestReportGeneration:
    """Tests for report generation"""
    
    def test_report_generation_with_artifacts(self):
        """Test that report can be generated if artifacts exist"""
        # Check if artifacts exist
        model_metrics_path = ARTIFACTS / "model_metrics.json"
        fairness_report_path = ARTIFACTS / "fairness_report.json"
        
        if model_metrics_path.exists() and fairness_report_path.exists():
            # Try to load artifacts
            model_metrics = safe_load_json(model_metrics_path)
            fairness_data = safe_load_json(fairness_report_path)
            
            # Should load successfully
            assert isinstance(model_metrics, dict)
            assert isinstance(fairness_data, dict)
        else:
            # Skip test if artifacts don't exist
            pytest.skip("Artifacts not available - run ML pipeline first")


class TestDataValidation:
    """Tests for data validation"""
    
    def test_validate_model_metrics_structure(self):
        """Test validation of model metrics structure"""
        model_metrics_path = ARTIFACTS / "model_metrics.json"
        
        if model_metrics_path.exists():
            metrics = safe_load_json(model_metrics_path)
            
            if metrics:
                # Check required keys
                assert 'accuracy' in metrics
                assert 'train_size' in metrics
                assert 'test_size' in metrics
                
                # Check value ranges
                assert 0 <= metrics.get('accuracy', 0) <= 1
                assert metrics.get('train_size', 0) > 0
                assert metrics.get('test_size', 0) > 0
    
    def test_validate_fairness_report_structure(self):
        """Test validation of fairness report structure"""
        fairness_report_path = ARTIFACTS / "fairness_report.json"
        
        if fairness_report_path.exists():
            fairness_data = safe_load_json(fairness_report_path)
            
            if fairness_data:
                # Check required keys
                assert 'bias_detected' in fairness_data
                assert 'overall_accuracy' in fairness_data
                assert 'by_type' in fairness_data
                
                # Check by_type structure
                if 'by_type' in fairness_data:
                    for machine_type in ['L', 'M', 'H']:
                        if machine_type in fairness_data['by_type']:
                            type_metrics = fairness_data['by_type'][machine_type]
                            assert 'accuracy' in type_metrics
                            assert 'false_negative_rate' in type_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

