"""
Main execution script for FBX Hedging Strategy Backtesting System
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config
from config.instruments import HEDGE_INSTRUMENTS
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from analysis.exposure_analyzer import ExposureAnalyzer
from analysis.hedge_optimizer import HedgeOptimizer
from backtesting.backtest_engine import BacktestEngine
from reporting.excel_reporter import ExcelReporter
from utils.helpers import setup_logging, create_directories

# ML imports (conditional based on config)
try:
    from ml.ml_predictor import MLPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

def main():
    """Main execution function"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting FBX Hedging Strategy Backtesting System")
    
    try:
        # Create necessary directories
        create_directories()
        
        # Initialize configuration
        config = Config()
        
        # Load and process data
        logger.info("Loading market data...")
        data_loader = DataLoader(config)
        raw_data = data_loader.load_all_data()
        
        data_processor = DataProcessor(config)
        processed_data = data_processor.process_data(raw_data)
        
        # Machine Learning Pipeline (if enabled)
        ml_results = None
        if config.ML_ENABLED and ML_AVAILABLE:
            logger.info("Running ML pipeline...")
            ml_predictor = MLPredictor(config)
            ml_results = ml_predictor.run_full_ml_pipeline(processed_data)
            
            if ml_results.get('status') == 'completed':
                logger.info("ML pipeline completed successfully")
                logger.info(f"Best model: {ml_results.get('best_single_model', 'N/A')}")
                logger.info(f"Best ensemble: {ml_results.get('best_ensemble', 'N/A')}")
            else:
                logger.warning("ML pipeline failed or incomplete")
        
        elif config.ML_ENABLED and not ML_AVAILABLE:
            logger.warning("ML is enabled but ML dependencies are not available")
        
        # Analyze exposure
        logger.info("Analyzing FBX exposure...")
        exposure_analyzer = ExposureAnalyzer(config)
        exposure_results = exposure_analyzer.analyze_exposure(processed_data)
        
        # Optimize hedge ratios
        logger.info("Optimizing hedge ratios...")
        hedge_optimizer = HedgeOptimizer(config)
        hedge_ratios = hedge_optimizer.optimize_hedge_ratios(processed_data, exposure_results)
        
        # Run backtesting
        logger.info("Running backtesting...")
        backtest_engine = BacktestEngine(config)
        backtest_results = backtest_engine.run_backtest(
            processed_data, 
            exposure_results, 
            hedge_ratios
        )
        
        # Generate reports
        logger.info("Generating Excel report...")
        excel_reporter = ExcelReporter(config)
        report_path = excel_reporter.generate_report(
            processed_data,
            exposure_results,
            hedge_ratios,
            backtest_results,
            ml_results=ml_results  # Pass ML results to reporter
        )
        
        logger.info(f"Backtesting completed successfully!")
        logger.info(f"Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("BACKTESTING SUMMARY")
        print("="*60)
        print(f"Data processed: {len(processed_data.get('fbx', pd.DataFrame()))} days")
        print(f"Hedge instruments: {len(processed_data.get('instruments', pd.DataFrame()).columns)}")
        
        if ml_results and ml_results.get('status') == 'completed':
            print(f"ML models trained: {ml_results.get('models_trained', 0)}")
            print(f"ML ensembles created: {ml_results.get('ensembles_created', 0)}")
            print(f"Best ML model: {ml_results.get('best_single_model', 'N/A')}")
            print(f"Best ensemble: {ml_results.get('best_ensemble', 'N/A')}")
        
        print(f"Report location: {report_path}")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
