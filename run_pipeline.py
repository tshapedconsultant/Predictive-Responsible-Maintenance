"""
Main script to run the complete ML pipeline and generate compliance report.
================================================================================
This is the entry point for the entire system. It orchestrates:
1. Running the ML pipeline (training model, generating explanations, etc.)
2. Generating the compliance report PDF

Simply run: python run_pipeline.py

Author: Predictive Maintenance Team
"""

import sys  # System operations (for exit codes)
import logging  # Logging system
from pathlib import Path  # File path handling

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    """
    Run complete pipeline: ML training -> Compliance report generation.
    
    This function:
    1. Runs the ML pipeline (ml_pipeline.py) which:
       - Loads and preprocesses data
       - Trains the Random Forest model
       - Generates LIME explanations
       - Performs drift detection
       - Analyzes fairness
    2. Generates the compliance report (make_compliance_report.py) which:
       - Loads all artifacts from step 1
       - Creates a professional PDF report
       - Includes all metrics and analyses
    
    All outputs are saved to the 'artifacts' folder and 'compliance_report.pdf'.
    """
    try:
        print("=" * 60)
        print("Predictive Maintenance - ML Pipeline & Compliance Report")
        print("=" * 60)
        print()
        
        # Step 1: Run ML pipeline
        print("Step 1: Running ML Pipeline...")
        print("-" * 60)
        try:
            from ml_pipeline import main as run_ml_pipeline
            run_ml_pipeline()
        except ImportError as e:
            print(f"ERROR: Error importing ML pipeline: {e}")
            print("Please ensure ml_pipeline.py exists and all dependencies are installed.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: ML Pipeline failed: {e}")
            logging.error(f"ML Pipeline error: {e}", exc_info=True)
            sys.exit(1)
        print()
        
        # Step 2: Generate compliance report
        print("Step 2: Generating Compliance Report...")
        print("-" * 60)
        try:
            from make_compliance_report import main as generate_report
            generate_report()
        except ImportError as e:
            print(f"ERROR: Error importing report generator: {e}")
            print("Please ensure make_compliance_report.py exists and all dependencies are installed.")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Report generation failed: {e}")
            logging.error(f"Report generation error: {e}", exc_info=True)
            sys.exit(1)
        print()
        
        print("=" * 60)
        print("SUCCESS: Pipeline completed successfully!")
        print("=" * 60)
        print(f"Check artifacts in: {Path(__file__).parent / 'artifacts'}")
        print(f"Compliance report: {Path(__file__).parent / 'compliance_report.pdf'}")
        
    except KeyboardInterrupt:
        print("\nWARNING: Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

