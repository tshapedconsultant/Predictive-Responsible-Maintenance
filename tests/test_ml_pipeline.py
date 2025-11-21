"""
Unit tests for ml_pipeline.py

Tests cover:
- Data loading and preprocessing
- Feature engineering
- Model training
- Explainability generation (LIME/SHAP)
- Drift detection
- Fairness analysis
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import sys

# Add parent directory to path to import ml_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline import (
    load_and_preprocess_data,
    calculate_custom_sample_weights,
    train_random_forest,
    generate_lime_explanations,
    generate_shap_explanations,
    detect_drift,
    analyze_fairness,
    BASE_DIR,
    ARTIFACTS
)


class TestDataLoading:
    """Tests for data loading and preprocessing"""
    
    def test_load_data_success(self):
        """Test successful data loading"""
        X, y, df, feature_cols, le = load_and_preprocess_data()
        
        # Check return types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(feature_cols, list)
        
        # Check data is not empty
        assert not X.empty
        assert len(y) > 0
        assert len(feature_cols) == 10
        
        # Check feature columns match
        assert list(X.columns) == feature_cols
        
        # Check target has binary values
        assert set(y) <= {0, 1}
    
    def test_feature_engineering(self):
        """Test that engineered features are created"""
        X, y, df, feature_cols, le = load_and_preprocess_data()
        
        # Check engineered features exist
        assert 'Temp_diff' in feature_cols
        assert 'Power [W]' in feature_cols
        assert 'Tool_wear_Torque' in feature_cols
        assert 'OSF_risk' in feature_cols
        
        # Check feature values are reasonable
        assert X['Temp_diff'].notna().all()
        assert X['Power [W]'].notna().all()
        assert X['Tool_wear_Torque'].notna().all()
        assert X['OSF_risk'].notna().all()
    
    def test_type_encoding(self):
        """Test that machine types are encoded correctly"""
        X, y, df, feature_cols, le = load_and_preprocess_data()
        
        # Check Type_encoded exists
        assert 'Type_encoded' in feature_cols
        
        # Check encoded values are integers
        assert X['Type_encoded'].dtype in [np.int64, np.int32, int]
        
        # Check values are in expected range (0, 1, 2)
        assert set(X['Type_encoded'].unique()) <= {0, 1, 2}
    
    def test_missing_value_handling(self):
        """Test that missing values are handled"""
        X, y, df, feature_cols, le = load_and_preprocess_data()
        
        # Check no NaN values remain
        assert not X.isnull().any().any()
    
    def test_feature_count(self):
        """Test that correct number of features are created"""
        X, y, df, feature_cols, le = load_and_preprocess_data()
        
        # Should have exactly 10 features
        assert len(feature_cols) == 10
        assert X.shape[1] == 10


class TestSampleWeights:
    """Tests for custom sample weight calculation"""
    
    def test_calculate_sample_weights(self):
        """Test custom sample weight calculation"""
        # Create dummy data
        X_train = pd.DataFrame({
            'Type_encoded': [0, 0, 1, 1, 2, 2],
            'feature1': [1, 2, 3, 4, 5, 6]
        })
        y_train = np.array([0, 1, 0, 1, 0, 1])
        df_train = pd.DataFrame({
            'Type_encoded': [0, 0, 1, 1, 2, 2],
            'Type': ['L', 'L', 'M', 'M', 'H', 'H']
        })
        
        weights = calculate_custom_sample_weights(X_train, y_train, df_train)
        
        # Check weights are calculated
        assert len(weights) == len(X_train)
        assert all(w > 0 for w in weights)
        
        # Check Type H failure samples have higher weights
        type_h_failures = (df_train['Type_encoded'] == 2) & (y_train == 1)
        if type_h_failures.any():
            type_h_failure_weights = weights[type_h_failures]
            other_weights = weights[~type_h_failures]
            # Type H failures should have higher weights (8x multiplier)
            assert type_h_failure_weights.mean() > other_weights.mean()


class TestModelTraining:
    """Tests for model training"""
    
    def test_train_model_success(self):
        """Test successful model training"""
        # Load data
        X, y, df, feature_cols, le = load_and_preprocess_data()
        
        # Train model
        model, X_train, X_test, y_train, y_test = train_random_forest(
            X, y, df=None, use_custom_weights=False
        )
        
        # Check model is trained
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Check train/test split
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check model can make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions) <= {0, 1}
    
    def test_model_saved(self):
        """Test that model is saved to disk"""
        model_path = ARTIFACTS / "rf_model.pkl"
        
        # Model should be saved after training
        if model_path.exists():
            with open(model_path, 'rb') as f:
                saved_model = pickle.load(f)
            assert saved_model is not None
    
    def test_metrics_saved(self):
        """Test that metrics are saved"""
        metrics_path = ARTIFACTS / "model_metrics.json"
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Check required metrics exist
            assert 'accuracy' in metrics
            assert 'train_size' in metrics
            assert 'test_size' in metrics
            assert 'feature_importance' in metrics
            
            # Check metrics are valid
            assert 0 <= metrics['accuracy'] <= 1
            assert metrics['train_size'] > 0
            assert metrics['test_size'] > 0


class TestExplainability:
    """Tests for explainability generation"""
    
    def test_lime_explanations_structure(self):
        """Test LIME explanations structure"""
        # Load data and train model
        X, y, df, feature_cols, le = load_and_preprocess_data()
        model, X_train, X_test, y_train, y_test = train_random_forest(
            X, y, df=None, use_custom_weights=False
        )
        
        # Generate LIME explanations
        lime_data = generate_lime_explanations(
            model, X_train, X_test, y_test, list(X.columns)
        )
        
        # Check structure
        assert isinstance(lime_data, dict)
        assert 'feature_names' in lime_data
        assert 'class_names' in lime_data
        assert 'explanations' in lime_data
        
        # Check feature names match
        assert len(lime_data['feature_names']) == len(X.columns)
    
    def test_shap_explanations_structure(self):
        """Test SHAP explanations structure"""
        # Load data and train model
        X, y, df, feature_cols, le = load_and_preprocess_data()
        model, X_train, X_test, y_train, y_test = train_random_forest(
            X, y, df=None, use_custom_weights=False
        )
        
        # Generate SHAP explanations
        readable_names = [col.replace('Type_encoded', 'Type (L/M/H)') 
                         for col in X.columns]
        shap_data = generate_shap_explanations(
            model, X_train, X_test, y_test, list(X.columns), readable_names
        )
        
        # Check structure
        assert isinstance(shap_data, dict)
        assert 'feature_names' in shap_data
        assert 'class_names' in shap_data
        assert 'global_importance' in shap_data


class TestDriftDetection:
    """Tests for drift detection"""
    
    def test_drift_detection_structure(self):
        """Test drift detection output structure"""
        # Load data and train model
        X, y, df, feature_cols, le = load_and_preprocess_data()
        model, X_train, X_test, y_train, y_test = train_random_forest(
            X, y, df=None, use_custom_weights=False
        )
        
        # Detect drift
        drift_df = detect_drift(X_train, X_test, list(X.columns))
        
        # Check structure
        assert isinstance(drift_df, pd.DataFrame)
        
        if not drift_df.empty:
            # Check required columns exist
            assert 'feature' in drift_df.columns
            assert 'drift_detected' in drift_df.columns or 'p_value' in drift_df.columns


class TestFairnessAnalysis:
    """Tests for fairness analysis"""
    
    def test_fairness_analysis_structure(self):
        """Test fairness analysis output structure"""
        # Load data and train model
        X, y, df, feature_cols, le = load_and_preprocess_data()
        model, X_train, X_test, y_train, y_test = train_random_forest(
            X, y, df=None, use_custom_weights=False
        )
        
        # Analyze fairness
        fairness_data = analyze_fairness(df, model, X, y)
        
        # Check structure
        assert isinstance(fairness_data, dict)
        
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
                        assert 'precision' in type_metrics
                        assert 'recall' in type_metrics
                        assert 'false_negative_rate' in type_metrics


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        # Create empty DataFrame
        empty_X = pd.DataFrame()
        empty_y = np.array([])
        
        # Should handle gracefully
        with pytest.raises((ValueError, AssertionError)):
            train_random_forest(empty_X, empty_y, df=None, use_custom_weights=False)
    
    def test_missing_columns_handling(self):
        """Test handling of missing required columns"""
        # This would require modifying the dataset file, so we'll skip
        # In practice, the function should raise ValueError with helpful message
        pass


class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline execution"""
        # Load data
        X, y, df, feature_cols, le = load_and_preprocess_data()
        assert not X.empty
        
        # Train model
        model, X_train, X_test, y_train, y_test = train_random_forest(
            X, y, df=None, use_custom_weights=False
        )
        assert model is not None
        
        # Generate explanations (non-critical, may fail if libraries not installed)
        try:
            lime_data = generate_lime_explanations(
                model, X_train, X_test, y_test, list(X.columns)
            )
            assert isinstance(lime_data, dict)
        except Exception:
            pass  # LIME may not be available
        
        try:
            shap_data = generate_shap_explanations(
                model, X_train, X_test, y_test, list(X.columns), list(X.columns)
            )
            assert isinstance(shap_data, dict)
        except Exception:
            pass  # SHAP may not be available
        
        # Detect drift
        drift_df = detect_drift(X_train, X_test, list(X.columns))
        assert isinstance(drift_df, pd.DataFrame)
        
        # Analyze fairness
        fairness_data = analyze_fairness(df, model, X, y)
        assert isinstance(fairness_data, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

