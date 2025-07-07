"""
Test script for ML functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config

# Test with minimal ML configuration
config = Config()

# Enable ML with minimal settings for testing
config.ML_ENABLED = True
config.ML_CONFIG['models'] = {
    'random_forest': True,
    'ridge': True,
    'lasso': False,
    'xgboost': False,  # Disable heavy models for testing
    'lstm': False,
    'svm': False,
    'elastic_net': False,
    'gradient_boosting': False,
    'extra_trees': False,
    'neural_network': False
}

config.ML_CONFIG['ensemble'] = {
    'voting': True,
    'stacking': False,  # Disable for testing
    'blending': False,
    'bayesian_optimization': False
}

config.ML_CONFIG['hyperopt']['n_trials'] = 10  # Reduce for testing
config.ML_CONFIG['hyperopt']['timeout'] = 300  # 5 minutes max

print("ML Configuration Test")
print("=" * 50)
print(f"ML Enabled: {config.ML_ENABLED}")
print(f"Enabled Models: {[k for k, v in config.ML_CONFIG['models'].items() if v]}")
print(f"Enabled Ensembles: {[k for k, v in config.ML_CONFIG['ensemble'].items() if v]}")
print(f"Hyperopt Trials: {config.ML_CONFIG['hyperopt']['n_trials']}")
print("=" * 50)

# Test imports
try:
    from ml.ml_predictor import MLPredictor
    print("✓ ML modules imported successfully")
except Exception as e:
    print(f"✗ ML import failed: {e}")

try:
    from ml.feature_engineer import FeatureEngineer
    print("✓ Feature Engineer imported successfully")
except Exception as e:
    print(f"✗ Feature Engineer import failed: {e}")

try:
    from ml.model_trainer import ModelTrainer
    print("✓ Model Trainer imported successfully")
except Exception as e:
    print(f"✗ Model Trainer import failed: {e}")

print("\nTest completed!")
