#!/usr/bin/env python3
"""
Deployment Verification Script
=============================

This script verifies that the FBX Hedging System is properly deployed
and all components are functioning correctly.
"""

import os
import sys
import importlib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_project_structure():
    """Verify that all required directories and files exist."""
    logger.info("🔍 Verifying project structure...")
    
    required_dirs = [
        'config', 'data', 'analysis', 'backtesting', 'reporting', 
        'ml', 'utils', 'tests', 'data_files', 'reports', 'logs', 'charts'
    ]
    
    required_files = [
        'main.py', 'requirements.txt', 'requirements-dev.txt', 
        'README.md', 'LICENSE', '.gitignore', '__version__.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
    
    if missing_dirs:
        logger.error(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    
    logger.info("✅ Project structure verified successfully")
    return True

def verify_python_modules():
    """Verify that all Python modules can be imported."""
    logger.info("🔍 Verifying Python modules...")
    
    modules_to_test = [
        'config.settings',
        'config.instruments',
        'data.data_loader',
        'data.data_processor',
        'analysis.exposure_analyzer',
        'analysis.hedge_optimizer',
        'backtesting.backtest_engine',
        'reporting.excel_reporter',
        'utils.helpers'
    ]
    
    # Test ML module separately since it might have optional dependencies
    ml_modules = [
        'ml.feature_engineer',
        'ml.model_trainer',
        'ml.ml_predictor'
    ]
    
    failed_imports = []
    ml_failed = []
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            logger.info(f"✅ {module_name} imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module_name}: {e}")
            failed_imports.append(module_name)
    
    for module_name in ml_modules:
        try:
            importlib.import_module(module_name)
            logger.info(f"✅ {module_name} imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Failed to import {module_name}: {e}")
            ml_failed.append(module_name)
    
    if failed_imports:
        logger.error(f"❌ Failed to import core modules: {failed_imports}")
        return False
    
    if ml_failed:
        logger.warning(f"⚠️ ML modules not available: {ml_failed}")
        logger.info("ℹ️ System will run with ML disabled")
    
    logger.info("✅ Core Python modules verified successfully")
    return True

def verify_dependencies():
    """Verify that all required dependencies are installed."""
    logger.info("🔍 Verifying dependencies...")
    
    core_dependencies = [
        'pandas', 'numpy', 'openpyxl', 'matplotlib', 'seaborn', 'joblib', 'tqdm'
    ]
    
    ml_dependencies = [
        'sklearn'  # scikit-learn imports as sklearn
    ]
    
    optional_dependencies = [
        'xgboost', 'optuna', 'shap', 'talib'
    ]
    
    missing_core = []
    missing_ml = []
    missing_optional = []
    
    for dep in core_dependencies:
        try:
            importlib.import_module(dep)
            logger.info(f"✅ {dep} available")
        except ImportError:
            missing_core.append(dep)
            logger.error(f"❌ {dep} not found")
    
    for dep in ml_dependencies:
        try:
            importlib.import_module(dep)
            logger.info(f"✅ {dep} available")
        except ImportError:
            missing_ml.append(dep)
            logger.warning(f"⚠️ {dep} not found (ML dependency)")
    
    for dep in optional_dependencies:
        try:
            importlib.import_module(dep)
            logger.info(f"✅ {dep} available (optional)")
        except ImportError:
            missing_optional.append(dep)
            logger.warning(f"⚠️ {dep} not found (optional)")
    
    if missing_core:
        logger.error(f"❌ Missing core dependencies: {missing_core}")
        return False
    
    if missing_ml:
        logger.warning(f"⚠️ Missing ML dependencies: {missing_ml}")
        logger.info("ℹ️ System will run with limited ML capabilities")
    
    if missing_optional:
        logger.info(f"ℹ️ Optional dependencies not found: {missing_optional}")
        logger.info("ℹ️ System will run with graceful degradation")
    
    logger.info("✅ Core dependencies verified successfully")
    return True

def verify_configuration():
    """Verify that configuration files are properly set up."""
    logger.info("🔍 Verifying configuration...")
    
    try:
        from config.settings import config
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"ℹ️ ML enabled: {config.ML_ENABLED}")
        logger.info(f"ℹ️ ML mode: {config.ML_MODE}")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False

def verify_data_generation():
    """Verify that mock data can be generated."""
    logger.info("🔍 Verifying data generation...")
    
    try:
        from config.settings import config
        from data.mock_data import MockDataGenerator
        
        generator = MockDataGenerator(config)
        data = generator.generate_market_data()
        
        if data is not None and len(data) > 0:
            logger.info("✅ Mock data generation successful")
            logger.info(f"ℹ️ Generated {len(data)} days of data")
            return True
        else:
            logger.error("❌ Mock data generation failed")
            return False
    except Exception as e:
        logger.error(f"❌ Data generation error: {e}")
        return False

def verify_reporting():
    """Verify that reporting system is working."""
    logger.info("🔍 Verifying reporting system...")
    
    try:
        from config.settings import config
        from reporting.excel_reporter import ExcelReporter
        
        reporter = ExcelReporter(config)
        logger.info("✅ Excel reporter initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Reporting system error: {e}")
        return False

def run_deployment_verification():
    """Run all verification checks."""
    logger.info("🚀 Starting deployment verification...")
    logger.info("=" * 50)
    
    checks = [
        ("Project Structure", verify_project_structure),
        ("Python Modules", verify_python_modules),
        ("Dependencies", verify_dependencies),
        ("Configuration", verify_configuration),
        ("Data Generation", verify_data_generation),
        ("Reporting System", verify_reporting)
    ]
    
    passed = 0
    failed = 0
    
    for check_name, check_func in checks:
        logger.info(f"\n📋 Running {check_name} verification...")
        try:
            if check_func():
                passed += 1
                logger.info(f"✅ {check_name} verification PASSED")
            else:
                failed += 1
                logger.error(f"❌ {check_name} verification FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {check_name} verification FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info("🎯 DEPLOYMENT VERIFICATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 ALL CHECKS PASSED - DEPLOYMENT READY!")
        logger.info("🚀 System is ready for production use")
        return True
    else:
        logger.error("⚠️ SOME CHECKS FAILED - REVIEW REQUIRED")
        logger.error("🔧 Please fix the issues before deployment")
        return False

if __name__ == "__main__":
    success = run_deployment_verification()
    sys.exit(0 if success else 1)
