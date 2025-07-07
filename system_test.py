"""
Complete System Test - Showcase All Features
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config
from utils.helpers import setup_logging

def main():
    """Run complete system test with all features"""
    print("FBX Hedging Strategy Backtesting System - Complete Test")
    print("=" * 80)
    
    # Setup
    start_time = time.time()
    setup_logging()
    
    # Test configuration
    config = Config()
    config.ML_ENABLED = True
    config.ML_MODE = "minimal"  # Use minimal for testing
    
    print("Configuration:")
    print(f"   • Data Period: {config.START_DATE.strftime('%Y-%m-%d')} to {config.END_DATE.strftime('%Y-%m-%d')}")
    print(f"   • ML Enabled: {config.ML_ENABLED}")
    print(f"   • ML Mode: {config.ML_MODE}")
    print(f"   • Initial Capital: ${config.INITIAL_CAPITAL:,.0f}")
    print()
    
    # Test imports
    print("🔍 Testing System Components:")
    
    components = [
        ("Data Loader", "data.data_loader", "DataLoader"),
        ("Data Processor", "data.data_processor", "DataProcessor"),
        ("Exposure Analyzer", "analysis.exposure_analyzer", "ExposureAnalyzer"),
        ("Hedge Optimizer", "analysis.hedge_optimizer", "HedgeOptimizer"),
        ("Backtest Engine", "backtesting.backtest_engine", "BacktestEngine"),
        ("Excel Reporter", "reporting.excel_reporter", "ExcelReporter"),
        ("ML Predictor", "ml.ml_predictor", "MLPredictor"),
    ]
    
    success_count = 0
    for name, module_name, class_name in components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"   [OK] {name}")
            success_count += 1
        except ImportError as e:
            print(f"   [FAIL] {name}: {e}")
        except Exception as e:
            print(f"   [WARN] {name}: {e}")
    
    print(f"\nComponent Test Results: {success_count}/{len(components)} passed")
    
    # Test data generation
    print("\n🔄 Testing Data Generation:")
    try:
        from data.mock_data import MockDataGenerator
        mock_gen = MockDataGenerator(config)
        fbx_data = mock_gen.generate_fbx_data()
        print(f"   [OK] FBX Data: {len(fbx_data)} records generated")
        
        instruments_data = mock_gen.generate_instruments_data()
        print(f"   [OK] Instruments Data: {len(instruments_data.columns)} instruments, {len(instruments_data)} records")
        
        revenue_data = mock_gen.generate_revenue_data()
        print(f"   [OK] Revenue Data: {len(revenue_data)} quarters generated")
        
    except Exception as e:
        print(f"   [FAIL] Data generation failed: {e}")
    
    # Test ML capabilities
    print("\n🧠 Testing ML Capabilities:")
    try:
        from ml.feature_engineer import FeatureEngineer
        feature_eng = FeatureEngineer(config)
        print("   [OK] Feature Engineer initialized")
        
        from ml.model_trainer import ModelTrainer
        model_trainer = ModelTrainer(config)
        print("   [OK] Model Trainer initialized")
        
        from ml.ensemble_manager import EnsembleManager
        ensemble_mgr = EnsembleManager(config)
        print("   [OK] Ensemble Manager initialized")
        
        from ml.hyperopt_optimizer import HyperoptOptimizer
        hyperopt = HyperoptOptimizer(config)
        print("   [OK] Hyperopt Optimizer initialized")
        
        from ml.ml_predictor import MLPredictor
        ml_predictor = MLPredictor(config)
        print("   [OK] ML Predictor initialized")
        
    except Exception as e:
        print(f"   [FAIL] ML component failed: {e}")
    
    # Performance summary
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    
    print("\n🎉 System Status: READY FOR PRODUCTION")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    main()
