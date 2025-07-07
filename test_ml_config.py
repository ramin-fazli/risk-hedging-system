"""
ML Configuration Test Script

This script tests different ML configurations and provides recommendations.
"""

import os
import sys
import time
import psutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ml_configuration():
    """Test ML configuration options"""
    print("="*60)
    print("FBX HEDGING STRATEGY - ML CONFIGURATION TEST")
    print("="*60)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # System information
    print("SYSTEM INFORMATION:")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Python version: {sys.version}")
    print()
    
    # Test configurations
    configurations = {
        "minimal": {
            "description": "Quick testing, minimal resources",
            "estimated_time": "5-15 minutes",
            "ram_requirement": "2-4 GB",
            "cpu_requirement": "2-4 cores"
        },
        "testing": {
            "description": "Development and validation",
            "estimated_time": "20-60 minutes",
            "ram_requirement": "4-8 GB",
            "cpu_requirement": "4-8 cores"
        },
        "production": {
            "description": "Full deployment, maximum accuracy",
            "estimated_time": "1-4 hours",
            "ram_requirement": "8-16 GB",
            "cpu_requirement": "8+ cores"
        }
    }
    
    # Test import capabilities
    print("DEPENDENCY CHECK:")
    dependencies = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "xgboost": "xgboost",
        "tensorflow": "tensorflow",
        "optuna": "optuna",
        "ta-lib": "talib",
        "shap": "shap",
        "lime": "lime"
    }
    
    available_deps = {}
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            available_deps[name] = True
            print(f"✓ {name}")
        except ImportError:
            available_deps[name] = False
            print(f"✗ {name} (optional)")
    
    print()
    
    # Recommendations
    print("CONFIGURATION RECOMMENDATIONS:")
    print("-" * 40)
    
    available_ram = psutil.virtual_memory().available / (1024**3)
    cpu_cores = psutil.cpu_count()
    
    for config_name, config_info in configurations.items():
        print(f"\n{config_name.upper()} MODE:")
        print(f"  Description: {config_info['description']}")
        print(f"  Estimated time: {config_info['estimated_time']}")
        print(f"  RAM requirement: {config_info['ram_requirement']}")
        print(f"  CPU requirement: {config_info['cpu_requirement']}")
        
        # Check if system meets requirements
        ram_min = float(config_info['ram_requirement'].split('-')[0])
        cpu_req = config_info['cpu_requirement'].split('-')[0].replace('+', '').replace(' cores', '')
        cpu_min = int(cpu_req)
        
        ram_ok = available_ram >= ram_min
        cpu_ok = cpu_cores >= cpu_min
        
        if ram_ok and cpu_ok:
            print("  Status: ✓ RECOMMENDED")
        else:
            print("  Status: ⚠ CHECK REQUIREMENTS")
            if not ram_ok:
                print("    - Insufficient RAM")
            if not cpu_ok:
                print("    - Insufficient CPU cores")
    
    print()
    
    # Best recommendation
    if available_ram >= 8 and cpu_cores >= 8:
        recommended = "production"
    elif available_ram >= 4 and cpu_cores >= 4:
        recommended = "testing"
    else:
        recommended = "minimal"
    
    print(f"RECOMMENDED CONFIGURATION: {recommended.upper()}")
    print()
    
    # Configuration setup
    print("CONFIGURATION SETUP:")
    print("To use the recommended configuration, set in config/settings.py:")
    print(f"config.ML_ENABLED = True")
    print(f"config.ML_MODE = '{recommended}'")
    print()
    
    # Missing dependencies
    missing_deps = [name for name, available in available_deps.items() if not available]
    if missing_deps:
        print("OPTIONAL DEPENDENCIES TO INSTALL:")
        for dep in missing_deps:
            if dep in ["xgboost", "tensorflow", "optuna", "ta-lib", "shap", "lime"]:
                print(f"  pip install {dep}")
        print()
    
    # Quick test
    print("QUICK FUNCTIONALITY TEST:")
    try:
        from config.settings import Config
        config = Config()
        config.ML_ENABLED = True
        config.ML_MODE = recommended
        print("✓ Configuration loaded successfully")
        
        # Test ML imports
        from ml.feature_engineer import FeatureEngineer
        from ml.model_trainer import ModelTrainer
        from ml.ml_predictor import MLPredictor
        print("✓ ML modules imported successfully")
        
        print("✓ System ready for ML-enhanced backtesting")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Please check dependencies and configuration")
    
    print()
    print("="*60)
    print("Test completed. Review recommendations above.")
    print("="*60)

if __name__ == "__main__":
    test_ml_configuration()
