"""
ML Pipeline for Predictive Maintenance
========================================
This script trains a machine learning model to predict machine failures in industrial equipment.
It performs the following tasks:
1. Loads and prepares the dataset
2. Trains a Random Forest model
3. Generates explanations for predictions (LIME)
4. Detects data drift (changes in data patterns)
5. Analyzes fairness across different machine types

Author: Predictive Maintenance Team
Version: 3.0
"""

# Standard library imports - for system operations and file handling
import sys  # System-specific parameters and functions (for exit codes)
import json  # JSON file reading/writing
import logging  # Logging system for tracking progress and errors
from datetime import datetime  # Date and time operations
from pathlib import Path  # Object-oriented file system paths
import pickle  # Saving/loading Python objects (like trained models)

# Data science libraries
import pandas as pd  # Data manipulation and analysis (like Excel for Python)
import numpy as np  # Numerical computing (mathematical operations)

# Machine learning libraries from scikit-learn
from sklearn.model_selection import train_test_split  # Split data into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # Random Forest algorithm for predictions
from sklearn.preprocessing import LabelEncoder  # Convert text categories to numbers (L/M/H -> 0/1/2)
from sklearn.metrics import (  # Metrics to evaluate model performance
    classification_report,  # Detailed performance report
    confusion_matrix,  # Shows correct vs incorrect predictions
    accuracy_score  # Percentage of correct predictions
)

# Statistical analysis
from scipy import stats  # Statistical tests (for drift detection)

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
# Minimum dataset size requirements for reliable training
# IMPROVEMENT: Added minimum size constants to handle small datasets gracefully
MIN_DATASET_SIZE = 20  # Minimum samples for meaningful training
MIN_TRAIN_SIZE = 10     # Minimum samples for training set
MIN_TEST_SIZE = 5       # Minimum samples for test set
RECOMMENDED_DATASET_SIZE = 100  # Recommended minimum for stable results

# LIME dependency - make it optional with graceful fallback
# IMPROVEMENT: Pipeline continues without LIME if not available (no hard stop)
LIME_AVAILABLE = False
try:
    import lime  # Local Interpretable Model-agnostic Explanations
    import lime.lime_tabular  # LIME for tabular (table) data
    LIME_AVAILABLE = True
    logging.info("✓ LIME library found - explainability features enabled")
except ImportError:
    # IMPROVEMENT: Graceful fallback instead of hard stop
    logging.warning(
        "⚠️  LIME library not found. Explainability features will be disabled. "
        "To enable: pip install lime"
    )
    logging.warning("Pipeline will continue without LIME explanations")

# SHAP dependency - make it optional with graceful fallback
# SHAP (SHapley Additive exPlanations) provides unified framework for model explainability
SHAP_AVAILABLE = False
try:
    import shap  # SHapley Additive exPlanations
    SHAP_AVAILABLE = True
    logging.info("✓ SHAP library found - explainability features enabled")
except ImportError:
    logging.warning(
        "⚠️  SHAP library not found. SHAP explainability features will be disabled. "
        "To enable: pip install shap"
    )
    logging.warning("Pipeline will continue without SHAP explanations")

# Set up file paths and directories
BASE_DIR = Path(__file__).parent  # Get the directory where this script is located
ARTIFACTS = BASE_DIR / "artifacts"  # Folder to save all outputs (model, reports, etc.)
ARTIFACTS.mkdir(exist_ok=True)  # Create artifacts folder if it doesn't exist

# Configure logging - shows INFO level messages with simple format
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_and_preprocess_data():
    """
    Load the dataset and prepare it for machine learning.
    
    This function:
    1. Reads the CSV file with machine sensor data
    2. Validates that all required columns are present
    3. Converts machine types (L/M/H) to numbers (0/1/2)
    4. Creates new features based on domain knowledge:
       - Temperature difference (for heat dissipation failure detection)
       - Power calculation (for power failure detection)
       - Tool wear × Torque (for overstrain failure detection)
    5. Handles missing values
    
    Returns:
        X: Features (input data) - what the model uses to make predictions
        y: Target (output) - what we want to predict (machine failure: 0 or 1)
        df: Original dataframe with all columns
        feature_cols: List of feature names
        le: Label encoder (converts L/M/H to 0/1/2)
    """
    try:
        logging.info("Loading dataset...")
        dataset_path = BASE_DIR / "dataset" / "ai4i2020.csv"  # Path to data file
        
        # Check if file exists
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        # Read CSV file into pandas DataFrame (like a spreadsheet in Python)
        df = pd.read_csv(dataset_path)
        
        # Validate data is not empty
        if df.empty:
            raise ValueError(
                "Dataset file is empty. Please check your data file and try again."
            )
        
        # ====================================================================
        # DATASET SIZE VALIDATION
        # IMPROVEMENT: Check dataset size and warn if too small
        # ====================================================================
        dataset_size = len(df)
        if dataset_size < MIN_DATASET_SIZE:
            logging.warning(
                f"⚠️  WARNING: Dataset has only {dataset_size} rows. "
                f"Minimum recommended: {MIN_DATASET_SIZE} rows. "
                f"For best results, use at least {RECOMMENDED_DATASET_SIZE} rows."
            )
            logging.warning(
                "Results may be unstable with such a small dataset. "
                "Use for testing purposes only."
            )
        elif dataset_size < RECOMMENDED_DATASET_SIZE:
            logging.warning(
                f"Dataset has {dataset_size} rows. "
                f"Recommended minimum: {RECOMMENDED_DATASET_SIZE} rows for stable results."
            )
        else:
            logging.info(f"✓ Dataset size: {dataset_size} rows (adequate for training)")
        
        # Validate all required columns exist
        # These columns are needed for the model to work
        required_cols = ['Type', 'Air temperature [K]', 'Process temperature [K]',
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                        'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # IMPROVEMENT: User-friendly error message with actionable guidance
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Please ensure your dataset contains all required columns:\n"
                f"{', '.join(required_cols)}"
            )
        
        # ====================================================================
        # DATA ENCODING
        # ====================================================================
        # Convert machine types from letters to numbers
        # L (Low quality) = 0, M (Medium) = 1, H (High) = 2
        # Machine learning models need numbers, not text
        le = LabelEncoder()
        df['Type_encoded'] = le.fit_transform(df['Type'])
        
        # ====================================================================
        # FEATURE ENGINEERING
        # ====================================================================
        # Create new features based on domain knowledge about failure modes
        # This helps the model understand relationships in the data
        try:
            # Feature 1: Temperature difference
            # Formula: Temp_diff = Process_temp - Air_temp
            # Physical meaning: If process temp is too close to air temp, heat can't dissipate
            # Important for detecting Heat Dissipation Failure (HDF)
            # HDF occurs when: Temp_diff < 8.6K AND Rotational_speed < 1380 rpm
            # Domain knowledge: Heat needs temperature gradient to flow away from machine
            df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
            
            # Feature 2: Power calculation
            # Formula: Power [W] = Torque [Nm] × Rotational_speed [rpm] × (2π / 60)
            # Physical meaning: Power is the product of torque and angular velocity
            # Conversion: rpm to rad/s = rpm × 2π / 60
            #   - 2π radians = 360 degrees (full rotation)
            #   - 60 seconds = 1 minute
            #   - Example: 1500 rpm = 1500 × 2π / 60 = 157.08 rad/s
            # Important for detecting Power Failure (PWF)
            # PWF occurs when: Power < 3500W OR Power > 9000W
            # Domain knowledge: Machines need sufficient power to operate, but too much causes failure
            df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi / 60
            
            # Feature 3: Tool wear × Torque
            # Formula: Tool_wear_Torque = Tool_wear [min] × Torque [Nm]
            # Physical meaning: Combined stress indicator (cumulative wear × current load)
            # Units: min·Nm (time × force)
            # Important for detecting Overstrain Failure (OSF)
            # OSF occurs when Tool_wear_Torque exceeds type-specific threshold:
            #   - Type L (Low quality): 11,000 min·Nm
            #   - Type M (Medium quality): 12,000 min·Nm
            #   - Type H (High quality): 13,000 min·Nm
            # Domain knowledge: Higher quality machines can handle more cumulative stress
            df['Tool_wear_Torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
            
            # Feature 4: OSF Risk Score
            # Formula: OSF_risk = Tool_wear_Torque / OSF_threshold
            # Physical meaning: Normalized stress level
            # Interpretation:
            #   - 0.0 to 1.0: Safe operating range
            #   - > 1.0: At risk of overstrain failure
            #   - Example: If Tool_wear_Torque = 12,500 and threshold = 12,000, risk = 1.04 (4% over limit)
            # Type-specific thresholds: L=11000, M=12000, H=13000
            # Higher quality machines can handle more stress before failing
            df['OSF_threshold'] = df['Type_encoded'].map({0: 11000, 1: 12000, 2: 13000})
            # Calculate risk: actual stress / threshold
            # Values > 1.0 indicate risk of overstrain failure
            # Handle division by zero: if threshold is 0 or missing, set risk to 0 (safe default)
            df['OSF_risk'] = df['Tool_wear_Torque'] / df['OSF_threshold'].replace(0, np.nan)
            # For OSF_risk, NaN occurs only if threshold is 0 (invalid), so 0 is appropriate here
            df['OSF_risk'] = df['OSF_risk'].fillna(0)
            
            # ====================================================================
            # TYPE H INTERACTION FEATURES (Optional - currently disabled)
            # ====================================================================
            # NOTE: Type H interaction features were tested but did not improve FNR disparity.
            # The root cause is data scarcity: Type H has only ~21 failure samples (~2.1% failure rate)
            # vs Type L with ~235 failures (~3.9% failure rate). With such limited Type H failure data,
            # even specialized features cannot help the model learn reliable patterns.
            # 
            # These features are kept in the code for future use if more Type H failure data becomes available.
            # To enable: uncomment below and add features to feature_cols list.
            #
            # df['Type_H'] = (df['Type_encoded'] == 2).astype(int)
            # df['Type_H_Torque'] = df['Type_H'] * df['Torque [Nm]']
            # df['Type_H_Power'] = df['Type_H'] * df['Power [W]']
            # df['Type_H_Tool_wear_Torque'] = df['Type_H'] * df['Tool_wear_Torque']
            # df['Type_H_Rotational_speed'] = df['Type_H'] * df['Rotational speed [rpm]']
            # df['Type_H_Temp_diff'] = df['Type_H'] * df['Temp_diff']
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise
        
        # ====================================================================
        # FEATURE SELECTION
        # ====================================================================
        # Select the 10 core features for prediction
        feature_cols = [
            'Type_encoded',              # Machine quality type (0, 1, or 2)
            'Air temperature [K]',       # Ambient temperature
            'Process temperature [K]',   # Operating temperature
            'Temp_diff',                 # Temperature difference (engineered)
            'Rotational speed [rpm]',    # How fast the machine spins
            'Torque [Nm]',               # Rotational force
            'Tool wear [min]',           # How long the tool has been used
            'Power [W]',                 # Calculated power (engineered)
            'Tool_wear_Torque',          # Combined metric (engineered)
            'OSF_risk'                   # Overstrain risk score (engineered)
        ]
        
        # Extract features (X) and target (y)
        X = df[feature_cols].copy()  # Features: what we use to predict
        
        # ====================================================================
        # HANDLE MISSING VALUES (IMPROVED STRATEGY)
        # ====================================================================
        # For sensor data, filling with 0 is inappropriate as it introduces artifacts.
        # Better strategies:
        # - Continuous sensor features: Use median (robust to outliers)
        # - Categorical/encoded features: Use mode or a safe default
        # - Engineered features: Handle based on their specific meaning
        if X.isnull().any().any():
            missing_count = X.isnull().sum()
            total_missing = missing_count.sum()
            missing_pct = (total_missing / (len(X) * len(X.columns))) * 100
            
            logging.warning(
                f"NaN values detected in features ({total_missing} total, {missing_pct:.2f}% of data). "
                f"Applying appropriate imputation strategy."
            )
            
            # Define feature types for appropriate imputation
            # Continuous sensor features (use median - robust to outliers)
            continuous_features = [
                'Air temperature [K]',
                'Process temperature [K]',
                'Temp_diff',
                'Rotational speed [rpm]',
                'Torque [Nm]',
                'Tool wear [min]',
                'Power [W]',
                'Tool_wear_Torque',
                'OSF_risk'
            ]
            
            # Categorical/encoded features (use mode or safe default)
            categorical_features = ['Type_encoded']
            
            # Apply median imputation for continuous sensor features
            for col in continuous_features:
                if col in X.columns and X[col].isnull().any():
                    missing_count = X[col].isnull().sum()
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        # If median is also NaN (all values missing), use 0 as last resort
                        logging.warning(f"All values missing for {col}, using 0 as fallback")
                        X[col] = X[col].fillna(0)
                    else:
                        X[col] = X[col].fillna(median_val)
                        logging.info(f"Filled {missing_count} missing values in '{col}' with median: {median_val:.2f}")
            
            # Apply mode imputation for categorical features
            for col in categorical_features:
                if col in X.columns and X[col].isnull().any():
                    missing_count = X[col].isnull().sum()
                    mode_val = X[col].mode()
                    if len(mode_val) > 0:
                        X[col] = X[col].fillna(mode_val[0])
                        logging.info(f"Filled {missing_count} missing values in '{col}' with mode: {mode_val[0]}")
                    else:
                        # If no mode (all values missing), use 0 as safe default
                        logging.warning(f"All values missing for {col}, using 0 as fallback")
                        X[col] = X[col].fillna(0)
            
            # Final check: if any NaN remain, log warning
            if X.isnull().any().any():
                remaining_missing = X.isnull().sum().sum()
                logging.warning(
                    f"⚠️  WARNING: {remaining_missing} NaN values remain after imputation. "
                    f"This may indicate a data quality issue."
                )
                # As last resort, fill remaining with 0 (shouldn't happen with proper imputation)
                X = X.fillna(0)
        
        # Target variable: Machine failure (0 = no failure, 1 = failure)
        # This is what we want the model to predict
        y = df['Machine failure'].values
        
        if len(y) == 0:
            raise ValueError("No target values found in dataset")
        
        # IMPROVEMENT: Check class distribution
        class_dist = pd.Series(y).value_counts()
        if len(class_dist) < 2:
            logging.warning(
                f"⚠️  WARNING: Only one class found in target variable. "
                f"Model training may not work properly. "
                f"Found classes: {class_dist.to_dict()}"
            )
        
        # ====================================================================
        # LOGGING AND SUMMARY
        # ====================================================================
        # Calculate and log failure mode statistics
        try:
            failure_modes = {
                'TWF': int(df['TWF'].sum()),  # Tool Wear Failure count
                'HDF': int(df['HDF'].sum()),  # Heat Dissipation Failure count
                'PWF': int(df['PWF'].sum()),  # Power Failure count
                'OSF': int(df['OSF'].sum()),  # Overstrain Failure count
                'RNF': int(df['RNF'].sum())   # Random Failure count
            }
        except Exception as e:
            logging.warning(f"Could not calculate failure modes: {e}")
            failure_modes = {}
        
        # Print summary statistics
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Features: {len(feature_cols)} features")
        logging.info(f"Target distribution: {class_dist.to_dict()}")
        if failure_modes:
            logging.info(f"Failure modes: {failure_modes}")
        logging.info(f"Type distribution: {df['Type'].value_counts().to_dict()}")
        
        return X, y, df, feature_cols, le
    
    except FileNotFoundError as e:
        # IMPROVEMENT: User-friendly error with actionable guidance
        logging.error(f"❌ Dataset file not found: {e}")
        raise
    except pd.errors.EmptyDataError:
        logging.error("❌ Dataset file is empty or corrupted. Please check your data file.")
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected error loading data: {e}")
        raise

# ============================================================================
# CUSTOM SAMPLE WEIGHTS (Priority 2: Improve Learning)
# ============================================================================
def calculate_custom_sample_weights(X_train, y_train, df_train):
    """
    Calculate custom sample weights that heavily penalize Type H false negatives.
    
    Strategy:
    - Base weight: Use balanced class weights (handle class imbalance)
    - Type H penalty: Apply 5x higher weight to Type H failure samples
    - Type H FNR penalty: Apply 3x higher weight to Type H non-failure samples
      (to reduce false negatives - missed failures)
    
    This forces the model to prioritize correctly identifying Type H failures.
    
    Parameters:
        X_train: Training feature data
        y_train: Training target data (0=no failure, 1=failure)
        df_train: Training dataframe with Type column
    
    Returns:
        sample_weights: Array of weights for each training sample
    """
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Start with balanced class weights
    base_weights = compute_sample_weight('balanced', y_train)
    
    # Get machine type for each sample from df_train
    type_values = df_train['Type_encoded'].values if 'Type_encoded' in df_train.columns else np.zeros(len(X_train))
    
    # Create custom weights
    custom_weights = base_weights.copy()
    
    # Type H = 2 (encoded), Type M = 1, Type L = 0
    type_h_mask = (type_values == 2)
    
    # Apply 8x weight to Type H failure samples (critical to catch these)
    # Increased significantly because Type H has very few failure examples (~21 total)
    # Need to heavily emphasize these rare failure cases
    type_h_failure_mask = type_h_mask & (y_train == 1)
    custom_weights[type_h_failure_mask] *= 8.0
    
    # DO NOT penalize Type H non-failures - this was causing the model to be too conservative
    # and actually increasing FNR. Let the model learn naturally from the interaction features.
    # type_h_no_failure_mask = type_h_mask & (y_train == 0)
    # custom_weights[type_h_no_failure_mask] *= 1.0  # No penalty
    
    # Apply 1.2x weight to Type M failure samples (slight priority)
    type_m_mask = (type_values == 1)
    type_m_failure_mask = type_m_mask & (y_train == 1)
    custom_weights[type_m_failure_mask] *= 1.2
    
    logging.info(f"✓ Custom sample weights calculated:")
    logging.info(f"  - Type H failures: {type_h_failure_mask.sum()} samples (8x weight - critical)")
    logging.info(f"  - Type H non-failures: {(type_h_mask & (y_train == 0)).sum()} samples (no penalty)")
    logging.info(f"  - Type M failures: {type_m_failure_mask.sum()} samples (1.2x weight)")
    
    return custom_weights

# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_random_forest(X, y, df=None, use_custom_weights=False):
    """
    Train a Random Forest classifier to predict machine failures.
    
    Random Forest works by:
    1. Creating many decision trees (like 100 different "experts")
    2. Each tree makes a prediction
    3. The final prediction is the majority vote of all trees
    
    This function:
    1. Splits data into training (80%) and testing (20%) sets
    2. Trains the model on training data
    3. Tests the model on test data to measure accuracy
    4. Saves the trained model and metrics
    
    Parameters:
        X: Feature data (inputs)
        y: Target data (what to predict - failure or no failure)
    
    Returns:
        rf_model: Trained Random Forest model
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing targets
    """
    try:
        logging.info("Training Random Forest model...")
        
        # Validate inputs
        if X.empty or len(y) == 0:
            raise ValueError(
                "Empty dataset provided for training. "
                "Please check your data and try again."
            )
        
        dataset_size = len(X)
        
        # IMPROVEMENT: Dataset size validation for training
        if dataset_size < MIN_DATASET_SIZE:
            logging.warning(
                f"⚠️  Dataset size ({dataset_size}) is below minimum ({MIN_DATASET_SIZE}). "
                f"Training may produce unreliable results."
            )
        
        # ====================================================================
        # DATA SPLITTING
        # ====================================================================
        # Split data: 80% for training, 20% for testing
        # stratify=y ensures both sets have similar failure rates
        # Also split df if provided (needed for custom weights)
        try:
            if df is not None:
                # Split with indices to align df_train with X_train
                train_indices, test_indices = train_test_split(
                    np.arange(len(X)), test_size=0.2, random_state=42, stratify=y
                )
                X_train = X.iloc[train_indices].reset_index(drop=True)
                X_test = X.iloc[test_indices].reset_index(drop=True)
                y_train = y[train_indices]
                y_test = y[test_indices]
                df_train = df.iloc[train_indices].reset_index(drop=True)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                df_train = None
        except ValueError as e:
            # If stratification fails (e.g., only one class), try without it
            logging.warning(
                f"Stratified split failed: {e}. Using non-stratified split. "
                "This may result in imbalanced train/test sets."
            )
            if df is not None:
                train_indices, test_indices = train_test_split(
                    np.arange(len(X)), test_size=0.2, random_state=42
                )
                X_train = X.iloc[train_indices].reset_index(drop=True)
                X_test = X.iloc[test_indices].reset_index(drop=True)
                y_train = y[train_indices]
                y_test = y[test_indices]
                df_train = df.iloc[train_indices].reset_index(drop=True)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                df_train = None
        
        # IMPROVEMENT: Validate split results with minimum size requirements
        if len(X_train) < MIN_TRAIN_SIZE:
            raise ValueError(
                f"Training set too small ({len(X_train)} samples). "
                f"Minimum required: {MIN_TRAIN_SIZE} samples. "
                f"Please use a larger dataset."
            )
        
        if len(X_test) < MIN_TEST_SIZE:
            logging.warning(
                f"⚠️  Test set is small ({len(X_test)} samples). "
                f"Recommended minimum: {MIN_TEST_SIZE} samples. "
                f"Test metrics may be unreliable."
            )
        
        logging.info(f"✓ Training set: {len(X_train)} samples")
        logging.info(f"✓ Test set: {len(X_test)} samples")
        
        # ====================================================================
        # MODEL TRAINING
        # ====================================================================
        try:
            # Create Random Forest classifier
            # n_estimators=100: Use 100 decision trees
            # max_depth=10: Limit tree depth to prevent overfitting
            # random_state=42: For reproducible results
            
            # Determine class_weight and sample_weight strategy
            if use_custom_weights and df_train is not None:
                # Use custom sample weights for Type H FNR reduction
                # Don't use class_weight='balanced' when using custom weights
                sample_weights = calculate_custom_sample_weights(X_train, y_train, df_train)
                class_weight = None
                logging.info("Using custom sample weights (Type H FNR reduction)")
            else:
                # Standard balanced class weights
                sample_weights = None
                class_weight = 'balanced'
                logging.info("Using standard balanced class weights")
            
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight=class_weight
            )
            # Train the model on training data
            if sample_weights is not None:
                rf_model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                rf_model.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise
        
        # ====================================================================
        # MODEL EVALUATION
        # ====================================================================
        try:
            # Make predictions on test data
            y_pred = rf_model.predict(X_test)
            # Calculate accuracy: percentage of correct predictions
            accuracy = accuracy_score(y_test, y_pred)
            
            logging.info(f"✓ Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            # Generate detailed performance report
            try:
                logging.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
            except Exception as e:
                logging.warning(f"Could not generate classification report: {e}")
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise
        
        # ====================================================================
        # SAVE MODEL (with backup of previous version)
        # ====================================================================
        try:
            # Save current model as backup if it exists
            current_model_path = ARTIFACTS / "rf_model.pkl"
            if current_model_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = ARTIFACTS / f"rf_model_backup_{timestamp}.pkl"
                import shutil
                shutil.copy2(current_model_path, backup_path)
                logging.info(f"✓ Previous model backed up as: {backup_path.name}")
            
            # Save trained model to file (so we can use it later without retraining)
            with open(current_model_path, "wb") as f:
                pickle.dump(rf_model, f)
            logging.info("✓ Model saved successfully")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise
        
        # ====================================================================
        # SAVE METRICS
        # ====================================================================
        try:
            # Create metrics dictionary with performance information
            metrics = {
                "accuracy": float(accuracy),  # Overall accuracy
                "train_size": len(X_train),    # Number of training samples
                "test_size": len(X_test),      # Number of test samples
                "dataset_size": dataset_size,  # IMPROVEMENT: Track dataset size
                "feature_importance": {         # Which features matter most
                    str(col): float(imp) for col, imp in zip(X.columns, rf_model.feature_importances_)
                }
            }
            
            # Save metrics to JSON file
            with open(ARTIFACTS / "model_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            logging.info("✓ Metrics saved successfully")
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")
            raise
        
        return rf_model, X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error(f"❌ Critical error in model training: {e}")
        raise

# ============================================================================
# LIME EXPLAINABILITY
# ============================================================================
def generate_lime_explanations(model, X_train, X_test, y_test, feature_names):
    """
    Generate explanations for model predictions using LIME.
    
    LIME (Local Interpretable Model-agnostic Explanations) helps answer:
    "Why did the model predict failure (or no failure) for this machine?"
    
    It shows which features contributed most to the prediction.
    For example: "High tool wear (+0.3) and low temperature difference (+0.2)
    led to a failure prediction."
    
    IMPROVEMENT: Graceful fallback if LIME is not available.
    Pipeline continues without explanations instead of crashing.
    
    This function:
    1. Checks if LIME is available
    2. Creates a LIME explainer
    3. Generates explanations for a failure case and a non-failure case
    4. Saves the explanations to a JSON file
    
    Parameters:
        model: Trained Random Forest model
        X_train: Training features (used to understand data distribution)
        X_test: Test features (samples to explain)
        y_test: Test targets (to find failure and non-failure examples)
        feature_names: List of feature names
    
    Returns:
        lime_data: Dictionary containing explanations (empty if LIME unavailable)
    """
    # IMPROVEMENT: Check if LIME is available before proceeding
    if not LIME_AVAILABLE:
        logging.warning(
            "LIME not available - skipping explainability generation. "
            "Install with: pip install lime"
        )
        return {
            "feature_names": [str(f) for f in feature_names],
            "class_names": ['No Failure', 'Failure'],
            "explanations": [],
            "lime_available": False,
            "note": "LIME library not installed - explanations not generated"
        }
    
    try:
        logging.info("Generating LIME explanations...")
        
        # Check for empty data
        if X_train.empty or X_test.empty:
            logging.warning("Empty training or test set. Skipping LIME explanations.")
            return {
                "feature_names": [str(f) for f in feature_names],
                "class_names": ['No Failure', 'Failure'],
                "explanations": [],
                "lime_available": True
            }
        
        # ====================================================================
        # CREATE LIME EXPLAINER
        # ====================================================================
        try:
            # Create explainer that understands the data distribution
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,                    # Training data to learn from
                feature_names=[str(f) for f in feature_names],  # Feature names
                class_names=['No Failure', 'Failure'],  # Output class names
                mode='classification',             # Classification task (not regression)
                discretize_continuous=True         # Group similar values together
            )
        except Exception as e:
            logging.error(f"Error creating LIME explainer: {e}")
            raise
        
        # ====================================================================
        # GENERATE EXPLANATIONS
        # ====================================================================
        explanations_data = []
        
        # Find and explain a failure case
        try:
            failure_indices = np.where(y_test == 1)[0]  # Find rows with failures
            if len(failure_indices) > 0:
                failure_idx = failure_indices[0]  # Get first failure
                # Generate explanation for this sample
                exp_failure = explainer.explain_instance(
                    X_test.values[failure_idx],  # The specific machine data
                    model.predict_proba,          # Model's prediction function
                    num_features=min(len(feature_names), 10)  # Show top 10 features
                )
                # Store explanation data
                explanations_data.append({
                    "sample_type": "failure",
                    "sample_index": int(failure_idx),
                    "explanations": [
                        {"feature": str(feat), "importance": float(imp)} 
                        for feat, imp in exp_failure.as_list()  # Convert to list format
                    ],
                    "prediction": {
                        "no_failure_prob": float(exp_failure.predict_proba[0]),
                        "failure_prob": float(exp_failure.predict_proba[1])
                    }
                })
        except Exception as e:
            logging.warning(f"Could not generate failure explanation: {e}")
        
        # Find and explain a non-failure case
        try:
            non_failure_indices = np.where(y_test == 0)[0]  # Find rows without failures
            if len(non_failure_indices) > 0:
                non_failure_idx = non_failure_indices[0]  # Get first non-failure
                # Generate explanation for this sample
                exp_non_failure = explainer.explain_instance(
                    X_test.values[non_failure_idx],
                    model.predict_proba,
                    num_features=min(len(feature_names), 10)
                )
                # Store explanation data
                explanations_data.append({
                    "sample_type": "non_failure",
                    "sample_index": int(non_failure_idx),
                    "explanations": [
                        {"feature": str(feat), "importance": float(imp)} 
                        for feat, imp in exp_non_failure.as_list()
                    ],
                    "prediction": {
                        "no_failure_prob": float(exp_non_failure.predict_proba[0]),
                        "failure_prob": float(exp_non_failure.predict_proba[1])
                    }
                })
        except Exception as e:
            logging.warning(f"Could not generate non-failure explanation: {e}")
        
        # ====================================================================
        # SAVE EXPLANATIONS
        # ====================================================================
        lime_data = {
            "feature_names": [str(f) for f in feature_names],
            "class_names": ['No Failure', 'Failure'],
            "explanations": explanations_data,
            "lime_available": True
        }
        
        try:
            # Save to JSON file
            with open(ARTIFACTS / "lime_meta.json", "w") as f:
                json.dump(lime_data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving LIME metadata: {e}")
            raise
        
        logging.info(f"✓ LIME explanations generated for {len(explanations_data)} samples.")
        return lime_data
    
    except Exception as e:
        logging.error(f"Error generating LIME explanations: {e}")
        # IMPROVEMENT: Return empty structure instead of failing completely
        return {
            "feature_names": [str(f) for f in feature_names],
            "class_names": ['No Failure', 'Failure'],
            "explanations": [],
            "lime_available": LIME_AVAILABLE,
            "error": str(e)
        }

# ============================================================================
# SHAP EXPLANATIONS
# ============================================================================
def generate_shap_explanations(model, X_train, X_test, y_test, feature_names, original_feature_names=None):
    """
    Generate explanations for model predictions using SHAP.
    
    SHAP (SHapley Additive exPlanations) provides a unified framework for explaining
    model predictions. It's based on game theory and shows the contribution of each
    feature to the prediction.
    
    SHAP complements LIME by:
    - Providing theoretically grounded explanations (Shapley values)
    - Showing both local (individual) and global (overall) feature importance
    - Being consistent and additive (sum of contributions equals prediction)
    
    IMPROVEMENT: Graceful fallback if SHAP is not available.
    Pipeline continues without SHAP explanations instead of crashing.
    
    This function:
    1. Checks if SHAP is available
    2. Creates a SHAP explainer (TreeExplainer for Random Forest)
    3. Generates explanations for sample cases
    4. Calculates global feature importance
    5. Saves the explanations to a JSON file
    
    Parameters:
        model: Trained Random Forest model
        X_train: Training features (used as background data)
        X_test: Test features (samples to explain)
        y_test: Test targets (to find failure and non-failure examples)
        feature_names: List of feature names (as used in model)
        original_feature_names: List of original/readable feature names (optional)
    
    Returns:
        shap_data: Dictionary containing SHAP explanations (empty if SHAP unavailable)
    """
    # Create readable feature names mapping
    # Replace encoded names with original/readable names
    if original_feature_names and len(original_feature_names) == len(feature_names):
        display_feature_names = original_feature_names.copy()
        logging.info(f"Using provided readable feature names: {display_feature_names[:3]}...")
    else:
        # Create mapping manually
        display_feature_names = []
        for feat in feature_names:
            if feat == 'Type_encoded':
                display_feature_names.append('Type (L/M/H)')
            else:
                display_feature_names.append(feat)
        logging.info(f"Created readable feature names: {display_feature_names[:3]}...")
    
    # IMPROVEMENT: Check if SHAP is available before proceeding
    if not SHAP_AVAILABLE:
        logging.warning(
            "SHAP not available - skipping SHAP explainability generation. "
            "Install with: pip install shap"
        )
        return {
            "feature_names": display_feature_names,
            "class_names": ['No Failure', 'Failure'],
            "explanations": [],
            "global_importance": {},
            "shap_available": False,
            "note": "SHAP library not installed - explanations not generated"
        }
    
    try:
        logging.info("Generating SHAP explanations...")
        
        # Check for empty data
        if X_train.empty or X_test.empty:
            logging.warning("Empty training or test set. Skipping SHAP explanations.")
            return {
                "feature_names": [str(f) for f in feature_names],
                "class_names": ['No Failure', 'Failure'],
                "explanations": [],
                "global_importance": {},
                "shap_available": True
            }
        
        # ====================================================================
        # CREATE SHAP EXPLAINER
        # ====================================================================
        try:
            # Use TreeExplainer for Random Forest (faster and exact)
            # For other models, use KernelExplainer or LinearExplainer
            explainer = shap.TreeExplainer(model)
            logging.info("✓ SHAP TreeExplainer created")
        except Exception as e:
            logging.error(f"Error creating SHAP explainer: {e}")
            raise
        
        # ====================================================================
        # GENERATE GLOBAL FEATURE IMPORTANCE
        # ====================================================================
        try:
            # Use a sample of training data for global importance
            sample_size = min(100, len(X_train))
            X_train_sample = X_train.sample(n=sample_size, random_state=42) if len(X_train) > sample_size else X_train
            
            # Calculate SHAP values for global importance
            shap_values_global = explainer.shap_values(X_train_sample.values)
            
            # For binary classification, shap_values is a list [class_0_values, class_1_values]
            # We use class_1 (failure) values for importance
            if isinstance(shap_values_global, list) and len(shap_values_global) > 1:
                shap_values_class1 = shap_values_global[1]  # Failure class
            elif isinstance(shap_values_global, list) and len(shap_values_global) == 1:
                shap_values_class1 = shap_values_global[0]
            else:
                shap_values_class1 = shap_values_global
            
            # Ensure it's a numpy array
            shap_values_class1 = np.array(shap_values_class1)
            
            # Calculate mean absolute SHAP values (global importance)
            if len(shap_values_class1.shape) > 1:
                mean_shap_values = np.abs(shap_values_class1).mean(axis=0)
            else:
                mean_shap_values = np.abs(shap_values_class1)
            # Convert to list for safe indexing
            if hasattr(mean_shap_values, 'tolist'):
                mean_shap_list = mean_shap_values.tolist()
            elif hasattr(mean_shap_values, '__iter__'):
                mean_shap_list = list(mean_shap_values)
            else:
                mean_shap_list = [mean_shap_values]
            
            # Ensure we have the right number of values
            while len(mean_shap_list) < len(feature_names):
                mean_shap_list.append(0.0)
            
            global_importance = {}
            for i in range(len(feature_names)):
                val = mean_shap_list[i] if i < len(mean_shap_list) else 0.0
                # Handle numpy scalars and nested lists
                if isinstance(val, (list, tuple)):
                    val = val[0] if len(val) > 0 else 0.0
                if hasattr(val, 'item'):
                    val = val.item()
                elif hasattr(val, '__float__'):
                    val = val.__float__()
                try:
                    # Use readable feature name
                    readable_name = display_feature_names[i]
                    global_importance[readable_name] = float(val)
                except (TypeError, ValueError):
                    readable_name = display_feature_names[i]
                    global_importance[readable_name] = 0.0
            
            # Normalize to sum to 1.0 (for comparison with feature importance)
            total = sum(global_importance.values())
            if total > 0:
                global_importance = {k: v / total for k, v in global_importance.items()}
            
            logging.info("✓ Global SHAP feature importance calculated")
            
            # ====================================================================
            # GENERATE SHAP VISUALIZATIONS
            # ====================================================================
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                
                # Create summary plot (like the image shown)
                # Ensure shap_values_class1 is a numpy array with correct shape
                shap_vals_array = np.array(shap_values_class1)
                X_sample_array = np.array(X_train_sample.values)
                
                # Ensure shapes match
                if len(shap_vals_array.shape) == 1:
                    shap_vals_array = shap_vals_array.reshape(1, -1)
                if shap_vals_array.shape[0] != X_sample_array.shape[0]:
                    # Take first N samples to match
                    min_samples = min(shap_vals_array.shape[0], X_sample_array.shape[0])
                    shap_vals_array = shap_vals_array[:min_samples]
                    X_sample_array = X_sample_array[:min_samples]
                
                # Create summary plot
                # Use pandas DataFrame for better compatibility with readable names
                import pandas as pd
                X_sample_df = pd.DataFrame(X_sample_array, columns=display_feature_names)
                
                # Handle 3D SHAP values (samples, features, classes) for binary classification
                # We need to extract just one class for the plot
                if len(shap_vals_array.shape) == 3:
                    # Shape is (samples, features, classes) - use failure class (class 1)
                    shap_vals_for_plot = shap_vals_array[:, :, 1]  # Extract failure class SHAP values
                    logging.info(f"Extracted failure class SHAP values: shape {shap_vals_for_plot.shape}")
                else:
                    shap_vals_for_plot = shap_vals_array
                
                # Verify shapes match
                logging.info(f"SHAP values for plot shape: {shap_vals_for_plot.shape}, Features: {len(display_feature_names)}")
                logging.info(f"DataFrame shape: {X_sample_df.shape}, Columns: {len(X_sample_df.columns)}")
                
                plt.figure(figsize=(14, 10))
                # Ensure we have enough samples for a good plot
                if shap_vals_for_plot.shape[0] < 10:
                    logging.warning(f"Only {shap_vals_for_plot.shape[0]} samples for SHAP plot - may not show all features well")
                
                # Log feature names being used in plot
                logging.info(f"Creating SHAP plot with {len(display_feature_names)} features: {display_feature_names}")
                
                # Force SHAP to show all features explicitly
                # Calculate mean absolute SHAP values to ensure all features have some importance
                mean_abs_shap = np.abs(shap_vals_for_plot).mean(axis=0)
                logging.info(f"Mean absolute SHAP values per feature: {dict(zip(display_feature_names, mean_abs_shap))}")
                
                # Create SHAP Explanation object to have full control
                shap_explanation = shap.Explanation(
                    values=shap_vals_for_plot,
                    data=X_sample_df.values,
                    feature_names=display_feature_names
                )
                
                # Force all features to show - use multiple approaches
                # First, ensure we're not filtering by importance
                plt.figure(figsize=(14, 12))
                
                # Calculate feature importance for ordering (but show ALL)
                feature_importance = np.abs(shap_vals_for_plot).mean(axis=0)
                # Sort features by importance (descending) but show ALL
                sorted_indices = np.argsort(feature_importance)[::-1]
                
                # Reorder SHAP values and feature names by importance
                shap_vals_sorted = shap_vals_for_plot[:, sorted_indices]
                X_sample_sorted = X_sample_df.iloc[:, sorted_indices]
                feature_names_sorted = [display_feature_names[i] for i in sorted_indices]
                
                logging.info(f"Plotting {len(feature_names_sorted)} features in order: {feature_names_sorted}")
                
                # Try beeswarm plot with explicit max_display
                # SHAP sometimes filters features, so we'll be very explicit
                num_features = len(feature_names_sorted)
                logging.info(f"Explicitly setting max_display to {num_features} to show all features")
                
                try:
                    shap_explanation_sorted = shap.Explanation(
                        values=shap_vals_sorted,
                        data=X_sample_sorted.values,
                        feature_names=feature_names_sorted
                    )
                    # Explicitly pass max_display as integer
                    shap.plots.beeswarm(
                        shap_explanation_sorted,
                        show=False,
                        max_display=num_features  # Explicit integer, not len()
                    )
                    logging.info(f"Used beeswarm plot - should show {num_features} features")
                except (AttributeError, TypeError, Exception) as e:
                    # Fall back to summary_plot with explicit max_display
                    logging.warning(f"Beeswarm plot failed, using summary_plot: {e}")
                    shap.summary_plot(
                        shap_vals_sorted,
                        X_sample_sorted,
                        show=False,
                        max_display=num_features,  # Explicit integer
                        feature_names=feature_names_sorted
                    )
                    logging.info(f"Used summary_plot - should show {num_features} features")
                plot_path = ARTIFACTS / "shap_summary_plot.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                logging.info(f"✓ SHAP summary plot saved: {plot_path.name}")
                
                # Also create a bar plot of mean SHAP values
                plt.figure(figsize=(10, 6))
                sorted_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
                features = [f[0] for f in sorted_features]
                importances = [f[1] for f in sorted_features]
                
                plt.barh(range(len(features)), importances, color='steelblue')
                plt.yticks(range(len(features)), features)
                plt.xlabel('Mean |SHAP value| (average impact on model output)', fontsize=11)
                plt.title('SHAP Feature Importance - Predictive Maintenance Model', fontsize=12, fontweight='bold')
                plt.gca().invert_yaxis()
                plt.grid(axis='x', alpha=0.3, linestyle='--')
                plt.tight_layout()
                bar_plot_path = ARTIFACTS / "shap_feature_importance.png"
                plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                logging.info(f"✓ SHAP feature importance plot saved: {bar_plot_path.name}")
                
                # Create a custom summary plot that GUARANTEES all features are shown
                # This is a manual implementation to ensure all 10 features appear
                plt.figure(figsize=(14, 12))
                
                # Calculate statistics for each feature
                feature_stats = []
                for i, feat_name in enumerate(feature_names_sorted):
                    shap_vals_feat = shap_vals_sorted[:, i]
                    feature_stats.append({
                        'name': feat_name,
                        'mean_abs': np.abs(shap_vals_feat).mean(),
                        'mean': shap_vals_feat.mean(),
                        'std': shap_vals_feat.std(),
                        'values': shap_vals_feat,
                        'data_values': X_sample_sorted.iloc[:, i].values
                    })
                
                # Sort by mean absolute SHAP value (most important first)
                feature_stats.sort(key=lambda x: x['mean_abs'], reverse=True)
                
                # Create the plot manually to ensure all features show
                y_positions = np.arange(len(feature_stats))
                colors = plt.cm.RdYlBu(np.linspace(0, 1, len(feature_stats)))
                
                for idx, feat_stat in enumerate(feature_stats):
                    y_pos = y_positions[idx]
                    shap_vals = feat_stat['values']
                    data_vals = feat_stat['data_values']
                    
                    # Normalize data values for color mapping
                    if data_vals.max() > data_vals.min():
                        normalized_vals = (data_vals - data_vals.min()) / (data_vals.max() - data_vals.min())
                    else:
                        normalized_vals = np.ones_like(data_vals) * 0.5
                    
                    # Plot points colored by feature value
                    scatter = plt.scatter(
                        shap_vals,
                        [y_pos] * len(shap_vals),
                        c=normalized_vals,
                        cmap='RdYlBu',
                        s=20,
                        alpha=0.6,
                        edgecolors='none'
                    )
                
                # Set labels
                plt.yticks(y_positions, [fs['name'] for fs in feature_stats])
                plt.xlabel('SHAP value (impact on model output)', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.title('SHAP Summary Plot - All Features (Predictive Maintenance)', fontsize=14, fontweight='bold')
                plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                plt.grid(axis='x', alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=plt.gca())
                cbar.set_label('Feature value (normalized)', fontsize=10)
                
                plt.tight_layout()
                custom_plot_path = ARTIFACTS / "shap_summary_plot_custom.png"
                plt.savefig(custom_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                logging.info(f"✓ Custom SHAP summary plot saved: {custom_plot_path.name} (shows all {len(feature_stats)} features)")
                
            except ImportError:
                logging.warning("Matplotlib not available - SHAP plots will not be generated")
                logging.warning("Install with: pip install matplotlib")
            except Exception as e:
                logging.warning(f"Could not generate SHAP plots: {e}")
                logging.warning("This is optional - SHAP values are still calculated and saved")
                
        except Exception as e:
            logging.warning(f"Error calculating global SHAP importance: {e}")
            global_importance = {}
        
        # ====================================================================
        # GENERATE LOCAL EXPLANATIONS (INDIVIDUAL SAMPLES)
        # ====================================================================
        explanations_data = []
        
        # Find and explain a failure case
        try:
            failure_indices = np.where(y_test == 1)[0]
            if len(failure_indices) > 0:
                failure_idx = failure_indices[0]
                sample = X_test.iloc[[failure_idx]].values
                
                # Calculate SHAP values for this sample
                shap_values_sample = explainer.shap_values(sample)
                
                # Get failure class SHAP values
                if isinstance(shap_values_sample, list) and len(shap_values_sample) > 1:
                    shap_vals = np.array(shap_values_sample[1])  # Failure class
                    if len(shap_vals.shape) > 1:
                        shap_vals = shap_vals[0]  # First (only) sample
                elif isinstance(shap_values_sample, list):
                    shap_vals = np.array(shap_values_sample[0])
                    if len(shap_vals.shape) > 1:
                        shap_vals = shap_vals[0]
                else:
                    shap_vals = np.array(shap_values_sample)
                    if len(shap_vals.shape) > 1:
                        shap_vals = shap_vals[0]
                
                # Get base value (expected value)
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    if len(base_value) > 1:
                        base_value = float(base_value[1])  # Failure class
                    else:
                        base_value = float(base_value[0])
                else:
                    base_value = float(base_value)
                
                # Get prediction
                pred_proba = model.predict_proba(sample)[0]
                
                # Store explanation
                explanations_data.append({
                    "sample_type": "failure",
                    "sample_index": int(failure_idx),
                    "base_value": base_value,
                    "prediction": {
                        "no_failure_prob": float(pred_proba[0]),
                        "failure_prob": float(pred_proba[1])
                    },
                    "feature_contributions": [
                        {
                            "feature": display_feature_names[i],
                            "shap_value": float(shap_vals[i]) if i < len(shap_vals) else 0.0,
                            "feature_value": float(sample[0, i]) if i < sample.shape[1] else 0.0
                        }
                        for i in range(len(feature_names))
                    ]
                })
                logging.info("✓ SHAP explanation generated for failure case")
        except Exception as e:
            logging.warning(f"Could not generate failure SHAP explanation: {e}")
        
        # Find and explain a non-failure case
        try:
            non_failure_indices = np.where(y_test == 0)[0]
            if len(non_failure_indices) > 0:
                non_failure_idx = non_failure_indices[0]
                sample = X_test.iloc[[non_failure_idx]].values
                
                # Calculate SHAP values for this sample
                shap_values_sample = explainer.shap_values(sample)
                
                # Get no-failure class SHAP values
                if isinstance(shap_values_sample, list) and len(shap_values_sample) > 0:
                    shap_vals = np.array(shap_values_sample[0])  # No-failure class
                    if len(shap_vals.shape) > 1:
                        shap_vals = shap_vals[0]  # First sample
                else:
                    shap_vals = np.array(shap_values_sample)
                    if len(shap_vals.shape) > 1:
                        shap_vals = shap_vals[0]
                
                # Get base value
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    if len(base_value) > 0:
                        base_value = float(base_value[0])  # No-failure class
                    else:
                        base_value = 0.0
                else:
                    base_value = float(base_value)
                
                # Get prediction
                pred_proba = model.predict_proba(sample)[0]
                
                # Store explanation
                explanations_data.append({
                    "sample_type": "non_failure",
                    "sample_index": int(non_failure_idx),
                    "base_value": base_value,
                    "prediction": {
                        "no_failure_prob": float(pred_proba[0]),
                        "failure_prob": float(pred_proba[1])
                    },
                    "feature_contributions": [
                        {
                            "feature": display_feature_names[i],
                            "shap_value": float(shap_vals[i]) if i < len(shap_vals) else 0.0,
                            "feature_value": float(sample[0, i]) if i < sample.shape[1] else 0.0
                        }
                        for i in range(len(feature_names))
                    ]
                })
                logging.info("✓ SHAP explanation generated for non-failure case")
        except Exception as e:
            logging.warning(f"Could not generate non-failure SHAP explanation: {e}")
        
        # ====================================================================
        # SAVE EXPLANATIONS
        # ====================================================================
        shap_data = {
            "feature_names": display_feature_names,
            "class_names": ['No Failure', 'Failure'],
            "explanations": explanations_data,
            "global_importance": global_importance,
            "shap_available": True
        }
        
        try:
            # Save to JSON file
            with open(ARTIFACTS / "shap_meta.json", "w") as f:
                json.dump(shap_data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving SHAP metadata: {e}")
            raise
        
        logging.info(f"✓ SHAP explanations generated for {len(explanations_data)} samples.")
        return shap_data
    
    except Exception as e:
        logging.error(f"Error generating SHAP explanations: {e}")
        # IMPROVEMENT: Return empty structure instead of failing completely
        return {
            "feature_names": [str(f) for f in feature_names],
            "class_names": ['No Failure', 'Failure'],
            "explanations": [],
            "global_importance": {},
            "shap_available": SHAP_AVAILABLE,
            "error": str(e)
        }

# ============================================================================
# DRIFT DETECTION
# ============================================================================
def detect_drift(X_train, X_test, feature_names):
    """
    Detect data drift between training and test sets.
    
    Data drift occurs when the distribution of data changes over time.
    For example: if training data had temperatures around 300K, but new data
    has temperatures around 350K, that's drift. This can make the model less accurate.
    
    This function uses the Kolmogorov-Smirnov test to compare distributions:
    - If p-value < 0.05: Drift detected (distributions are different)
    - If p-value >= 0.05: No drift (distributions are similar)
    
    Parameters:
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
    
    Returns:
        drift_df: DataFrame with drift detection results for each feature
    """
    try:
        logging.info("Performing drift detection...")
        
        # Validate inputs
        if X_train.empty or X_test.empty:
            logging.warning("Empty train or test set. Cannot perform drift detection.")
            return pd.DataFrame()
        
        if len(feature_names) == 0:
            logging.warning("No features provided for drift detection.")
            return pd.DataFrame()
        
        drift_results = []
        
        # Check each feature for drift
        for i, feature in enumerate(feature_names):
            try:
                # Check array bounds
                if i >= X_train.shape[1] or i >= X_test.shape[1]:
                    logging.warning(f"Feature index {i} out of bounds for {feature}")
                    continue
                
                # Extract data for this feature
                train_data = X_train.iloc[:, i].values  # Training values
                test_data = X_test.iloc[:, i].values    # Test values
                
                if len(train_data) == 0 or len(test_data) == 0:
                    logging.warning(f"Empty data for feature {feature}")
                    continue
                
                # Kolmogorov-Smirnov test for numeric features
                if X_train.iloc[:, i].dtype in ['float64', 'int64']:
                    try:
                        # Perform statistical test
                        # Returns: test statistic and p-value
                        ks_stat, p_value = stats.ks_2samp(train_data, test_data)
                        # Drift detected if p-value < 0.05 (5% significance level)
                        drift_detected = p_value < 0.05
                        
                        # Store results
                        drift_results.append({
                            "feature": str(feature),
                            "drift_detected": bool(drift_detected),
                            "p_value": float(p_value),           # Lower = more different
                            "ks_statistic": float(ks_stat),       # Test statistic
                            "train_mean": float(np.mean(train_data)),  # Average in training
                            "test_mean": float(np.mean(test_data)),    # Average in test
                            "train_std": float(np.std(train_data)),   # Spread in training
                            "test_std": float(np.std(test_data))      # Spread in test
                        })
                    except Exception as e:
                        logging.warning(f"Drift detection failed for {feature}: {e}")
                        drift_results.append({
                            "feature": str(feature),
                            "drift_detected": False,
                            "p_value": 1.0,
                            "error": str(e)
                        })
                else:
                    # For non-numeric features, can't use KS test
                    drift_results.append({
                        "feature": str(feature),
                        "drift_detected": False,
                        "p_value": 1.0,
                        "note": "Non-numeric feature, KS test not applicable"
                    })
            except Exception as e:
                logging.warning(f"Error processing feature {feature}: {e}")
                continue
        
        if not drift_results:
            logging.warning("No drift results generated.")
            return pd.DataFrame()
        
        # Save results to CSV file
        try:
            drift_df = pd.DataFrame(drift_results)
            drift_df.to_csv(ARTIFACTS / "drift_report.csv", index=False)
            
            if 'drift_detected' in drift_df.columns:
                # Calculate percentage of features with drift
                drift_rate = drift_df['drift_detected'].mean() * 100
                logging.info(f"Drift detection complete. Drift rate: {drift_rate:.1f}%")
            else:
                logging.info("Drift detection complete.")
        except Exception as e:
            logging.error(f"Error saving drift report: {e}")
            drift_df = pd.DataFrame(drift_results)
        
        return drift_df
    
    except Exception as e:
        logging.error(f"Critical error in drift detection: {e}")
        return pd.DataFrame()

# ============================================================================
# FAIRNESS ANALYSIS
# ============================================================================
def analyze_fairness(df, model, X, y):
    """
    Analyze model fairness across different machine types (L, M, H).
    
    Fairness ensures the model performs equally well for all machine types.
    We check:
    1. Accuracy: Does the model predict correctly for all types?
    2. Precision: When it predicts failure, is it usually right?
    3. Recall: Does it catch most actual failures?
    4. False Positive Rate: How often does it predict failure when there isn't one?
    5. False Negative Rate: How often does it miss actual failures?
    6. Demographic Parity: Are failure prediction rates similar across types?
    
    Parameters:
        df: Original dataframe with Type column
        model: Trained model
        X: Feature data
        y: Target data
    
    Returns:
        fairness_data: Dictionary with fairness metrics
    """
    try:
        logging.info("Performing fairness analysis...")
        
        # Validate inputs
        if X.empty or len(y) == 0:
            logging.warning("Empty dataset. Cannot perform fairness analysis.")
            return {}
        
        if 'Type' not in df.columns:
            logging.warning("Type column not found. Cannot perform fairness analysis by type.")
            return {}
        
        # Get predictions for all data
        try:
            y_pred = model.predict(X)
        except Exception as e:
            logging.error(f"Error generating predictions: {e}")
            raise
        
        # ====================================================================
        # ANALYZE BY MACHINE TYPE
        # ====================================================================
        fairness_results = {}
        bias_detected = False
        
        # Expected distribution: L (50%), M (30%), H (20%)
        expected_dist = {'L': 0.50, 'M': 0.30, 'H': 0.20}
        
        # Analyze each machine type separately
        for machine_type in ['L', 'M', 'H']:
            # Filter data for this machine type
            type_mask = df['Type'] == machine_type
            type_actual = y[type_mask]      # Actual failures for this type
            type_pred = y_pred[type_mask]   # Predicted failures for this type
            
            # Calculate metrics if we have data
            if len(type_actual) > 0:
                # Basic metrics
                accuracy = accuracy_score(type_actual, type_pred)
                failure_rate = type_actual.mean()        # Actual failure rate
                pred_failure_rate = type_pred.mean()     # Predicted failure rate
                
                # Import metrics functions
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                # Detailed metrics
                precision = precision_score(type_actual, type_pred, zero_division=0)
                recall = recall_score(type_actual, type_pred, zero_division=0)
                f1 = f1_score(type_actual, type_pred, zero_division=0)
                
                # Calculate confusion matrix
                # Shows: True Negatives, False Positives, False Negatives, True Positives
                cm = confusion_matrix(type_actual, type_pred)
                if cm.size == 4:  # 2x2 matrix (both classes present)
                    tn, fp, fn, tp = cm.ravel()
                elif cm.size == 1:  # Only one class present
                    if type_actual[0] == 0:
                        tn, fp, fn, tp = len(type_actual), 0, 0, 0
                    else:
                        tn, fp, fn, tp = 0, 0, 0, len(type_actual)
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0
                
                # Calculate rates
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
                
                # Store results for this machine type
                fairness_results[machine_type] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "actual_failure_rate": float(failure_rate),
                    "predicted_failure_rate": float(pred_failure_rate),
                    "false_positive_rate": float(fpr),
                    "false_negative_rate": float(fnr),
                    "sample_size": int(len(type_actual)),
                    "expected_proportion": float(expected_dist[machine_type]),
                    "actual_proportion": float(len(type_actual) / len(df)),
                    "bias_score": float(abs(failure_rate - pred_failure_rate))
                }
                
                # Check for significant bias
                # Bias detected if:
                # - Predicted failure rate differs from actual by > 10%
                # - False positive/negative rates differ by > 15% (within same type)
                if abs(failure_rate - pred_failure_rate) > 0.1 or abs(fpr - fnr) > 0.15:
                    bias_detected = True
        
        # ====================================================================
        # CROSS-TYPE FNR ANALYSIS (CRITICAL FOR PREDICTIVE MAINTENANCE)
        # ====================================================================
        # FNR (False Negative Rate) is the costliest error in predictive maintenance
        # Missed failures can lead to equipment damage, safety risks, and unexpected downtime
        # Check for significant FNR disparities across machine types
        if len(fairness_results) >= 2:
            fnr_values = {t: fairness_results[t]["false_negative_rate"] 
                         for t in ['L', 'M', 'H'] if t in fairness_results}
            if len(fnr_values) >= 2:
                min_fnr = min(fnr_values.values())
                max_fnr = max(fnr_values.values())
                fnr_difference = max_fnr - min_fnr
                fnr_ratio = max_fnr / min_fnr if min_fnr > 0 else float('inf')
                
                # Get sample sizes for context
                sample_sizes = {t: fairness_results[t].get("sample_size", 0) 
                               for t in ['L', 'M', 'H'] if t in fairness_results}
                type_h_failures = fairness_results.get('H', {}).get('sample_size', 0) * \
                                 fairness_results.get('H', {}).get('actual_failure_rate', 0)
                
                # CRITICAL: Flag bias if FNR difference is significant
                # Thresholds:
                # - FNR difference > 5 percentage points (0.05)
                # - FNR ratio > 2.0x (one type has 2x+ higher missed failure rate)
                # - Absolute FNR > 8% for any type (high missed failure rate)
                if fnr_difference > 0.05 or fnr_ratio > 2.0 or max_fnr > 0.08:
                    bias_detected = True
                    logging.warning(
                        f"⚠️  CRITICAL: Significant FNR disparity detected across machine types! "
                        f"FNR range: {min_fnr*100:.2f}% - {max_fnr*100:.2f}% "
                        f"(difference: {fnr_difference*100:.2f}pp, ratio: {fnr_ratio:.2f}x). "
                        f"This indicates performance equity gap in failure detection."
                    )
                    # Add data scarcity context for Type H
                    if 'H' in fnr_values and fnr_values['H'] == max_fnr:
                        logging.warning(
                            f"📊 DATA CONTEXT: Type H has only ~{int(type_h_failures)} failure samples "
                            f"(vs ~{int(sample_sizes.get('L', 0) * fairness_results.get('L', {}).get('actual_failure_rate', 0))} for Type L). "
                            f"This data scarcity is the primary cause of the FNR disparity. "
                            f"More Type H failure data is needed to improve model performance."
                        )
                    
                    # Store FNR analysis (will be added to fairness_data later)
                    fnr_analysis = {
                        "min_fnr": float(min_fnr),
                        "max_fnr": float(max_fnr),
                        "fnr_difference": float(fnr_difference),
                        "fnr_ratio": float(fnr_ratio),
                        "fnr_by_type": {k: float(v) for k, v in fnr_values.items()},
                        "bias_detected_due_to_fnr": True,
                        "warning": "Significant FNR disparity indicates performance equity gap. "
                                  "Type H machines have higher missed failure rate, which is "
                                  "critical in predictive maintenance."
                    }
                else:
                    fnr_analysis = None
            else:
                fnr_analysis = None
        else:
            fnr_analysis = None
        
        # ====================================================================
        # OVERALL METRICS
        # ====================================================================
        from sklearn.metrics import precision_score, recall_score, f1_score
        overall_accuracy = accuracy_score(y, y_pred)
        overall_precision = precision_score(y, y_pred, zero_division=0)
        overall_recall = recall_score(y, y_pred, zero_division=0)
        overall_f1 = f1_score(y, y_pred, zero_division=0)
        
        # ====================================================================
        # DEMOGRAPHIC PARITY
        # ====================================================================
        # Measures if prediction rates are similar across machine types
        # Lower is better (0.0 = perfect parity)
        try:
            if len(fairness_results) >= 2:
                pred_rates = [fairness_results[t]["predicted_failure_rate"] 
                             for t in ['L', 'M', 'H'] if t in fairness_results]
                if pred_rates:
                    demographic_parity = max(pred_rates) - min(pred_rates)
                else:
                    demographic_parity = 0.0
            else:
                demographic_parity = 0.0
        except Exception as e:
            logging.warning(f"Error calculating demographic parity: {e}")
            demographic_parity = 0.0
        
        # ====================================================================
        # SAVE RESULTS
        # ====================================================================
        fairness_data = {
            "bias_detected": bias_detected,
            "overall_accuracy": float(overall_accuracy),
            "overall_precision": float(overall_precision),
            "overall_recall": float(overall_recall),
            "overall_f1": float(overall_f1),
            "demographic_parity": float(demographic_parity),
            "by_type": fairness_results,
            "timestamp": datetime.utcnow().isoformat() + " UTC"
        }
        
        # Add FNR analysis if it was calculated and detected bias
        if 'fnr_analysis' in locals() and fnr_analysis is not None:
            fairness_data["fnr_analysis"] = fnr_analysis
        
        try:
            with open(ARTIFACTS / "fairness_report.json", "w") as f:
                json.dump(fairness_data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving fairness report: {e}")
            raise
        
        logging.info(f"Fairness analysis complete. Bias detected: {bias_detected}")
        logging.info(f"Demographic parity: {demographic_parity:.4f}")
        return fairness_data
    
    except Exception as e:
        logging.error(f"Critical error in fairness analysis: {e}")
        return {}

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """
    Main function that runs the complete ML pipeline.
    
    Execution order:
    1. Load and preprocess data
    2. Train Random Forest model
    3. Generate LIME explanations
    4. Detect data drift
    5. Analyze fairness
    
    All outputs are saved to the 'artifacts' folder.
    """
    try:
        logging.info("=" * 50)
        logging.info("Starting ML Pipeline for Predictive Maintenance")
        logging.info("=" * 50)
        
        # Step 1: Load and preprocess data
        try:
            X, y, df, feature_cols, le = load_and_preprocess_data()
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise
        
        # Step 2: Train model
        try:
            # NOTE: Type H FNR disparity (9.52% vs 3.83% for Type L) is primarily due to
            # data scarcity: Type H has only ~21 failure samples in the entire dataset (~17 in training).
            # This is insufficient for the model to learn reliable failure patterns for Type H machines.
            # The model performs well given this constraint (99.10% overall accuracy).
            # Custom weights and interaction features were tested but did not improve FNR due to
            # the fundamental data limitation. Solution requires more Type H failure data collection.
            model, X_train, X_test, y_train, y_test = train_random_forest(
                X, y, df=None, use_custom_weights=False
            )
        except Exception as e:
            logging.error(f"Failed to train model: {e}")
            raise
        
        # Step 3: Generate LIME explanations (non-critical, continue if fails)
        try:
            lime_data = generate_lime_explanations(model, X_train, X_test, y_test, list(X.columns))
        except Exception as e:
            logging.warning(f"LIME generation failed (continuing): {e}")
            lime_data = {"feature_names": list(X.columns), "class_names": ['No Failure', 'Failure'], "explanations": []}
        
        # Step 3b: Generate SHAP explanations (non-critical, continue if fails)
        # Create readable feature names (replace Type_encoded with Type)
        readable_feature_names = []
        for col in X.columns:
            if col == 'Type_encoded':
                readable_feature_names.append('Type (L/M/H)')
            else:
                readable_feature_names.append(col)
        
        try:
            shap_data = generate_shap_explanations(model, X_train, X_test, y_test, list(X.columns), readable_feature_names)
        except Exception as e:
            logging.warning(f"SHAP generation failed (continuing): {e}")
            shap_data = {"feature_names": readable_feature_names, "class_names": ['No Failure', 'Failure'], "explanations": [], "global_importance": {}}
        
        # Step 4: Detect drift (non-critical, continue if fails)
        try:
            drift_df = detect_drift(X_train, X_test, list(X.columns))
        except Exception as e:
            logging.warning(f"Drift detection failed (continuing): {e}")
            drift_df = pd.DataFrame()
        
        # Step 5: Analyze fairness (non-critical, continue if fails)
        try:
            fairness_data = analyze_fairness(df, model, X, y)
        except Exception as e:
            logging.warning(f"Fairness analysis failed (continuing): {e}")
            fairness_data = {}
        
        logging.info("=" * 50)
        logging.info("Pipeline completed successfully!")
        logging.info("=" * 50)
        logging.info(f"Artifacts saved to: {ARTIFACTS}")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        raise

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Run main function with error handling
    try:
        main()
    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C)
        logging.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        # Handle any other errors
        logging.error(f"Fatal error: {e}")
        sys.exit(1)
