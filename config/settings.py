"""
Configuration settings for the FBX Hedging Strategy Backtesting System
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

class Config:
    """Main configuration class"""
    
    def __init__(self):
        # Project paths
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "data_files")
        self.REPORTS_DIR = os.path.join(self.PROJECT_ROOT, "reports")
        
        # Data configuration
        self.START_DATE = datetime(2020, 1, 1)
        self.END_DATE = datetime(2024, 12, 31)
        self.FREQUENCY = "D"  # Daily data
        
        # FBX configuration
        self.FBX_SYMBOL = "FBX"  # This will be mock data or external API
        self.FBX_BASE_VALUE = 1000  # Base value for FBX index
        
        # Revenue configuration
        self.REVENUE_FREQUENCY = "Q"  # Quarterly revenue data
        self.REVENUE_BASE = 100_000_000  # Base revenue in USD
        self.FBX_REVENUE_SENSITIVITY = 0.8  # Revenue sensitivity to FBX (beta)
        
        # Backtesting configuration
        self.INITIAL_CAPITAL = 10_000_000  # Initial portfolio value
        self.TRANSACTION_COST = 0.001  # 0.1% transaction cost
        self.REBALANCE_FREQUENCY = "M"  # Monthly rebalancing
        
        # Risk parameters
        self.MAX_POSITION_SIZE = 0.3  # Maximum 30% position in any single hedge
        self.VAR_CONFIDENCE = 0.95  # 95% confidence level for VaR
        self.LOOKBACK_PERIOD = 252  # 1 year lookback for calculations
        
        # Reporting configuration
        self.REPORT_NAME = f"FBX_Hedging_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        self.INCLUDE_CHARTS = True
        self.CHART_HEIGHT = 400
        self.CHART_WIDTH = 800
        
        # Hedge optimization methods
        self.HEDGE_METHODS = [
            "ols_regression",
            "minimum_variance",
            "correlation_based",
            "dynamic_beta"
        ]
        
        # Machine Learning Configuration
        self.ML_ENABLED = True  # Enable/disable ML functionality
        self.ML_MODE = "minimal"  # Options: "minimal", "testing", "production", "custom"
        self.ML_CONFIG = {
            # Feature Engineering
            "feature_engineering": {
                "technical_indicators": True,
                "rolling_statistics": True,
                "lag_features": True,
                "interaction_features": True,
                "fourier_features": True,
                "wavelet_features": False,  # Computationally intensive
                "pca_features": True,
                "polynomial_features": True
            },
            
            # Models to use
            "models": {
                "random_forest": True,
                "xgboost": True,
                "lstm": True,
                "transformer": False,  # Very computationally intensive
                "svm": True,
                "elastic_net": True,
                "ridge": True,
                "lasso": True,
                "gradient_boosting": True,
                "extra_trees": True,
                "neural_network": True
            },
            
            # Ensemble methods
            "ensemble": {
                "voting": True,
                "stacking": True,
                "blending": True,
                "bayesian_optimization": True
            },
            
            # Training parameters
            "training": {
                "train_split": 0.8,
                "validation_split": 0.1,
                "test_split": 0.1,
                "cross_validation_folds": 5,
                "time_series_split": True,
                "walk_forward_validation": True,
                "lookback_window": 252,
                "prediction_horizon": [1, 5, 10, 22],  # 1d, 1w, 2w, 1m
                "max_iterations": 1000,
                "early_stopping": True,
                "patience": 50
            },
            
            # Hyperparameter optimization
            "hyperopt": {
                "method": "optuna",  # optuna, hyperopt, or grid_search
                "n_trials": 100,
                "parallel": True,
                "pruning": True,
                "timeout": 3600  # 1 hour timeout
            },
            
            # Feature selection
            "feature_selection": {
                "method": "recursive_feature_elimination",
                "max_features": 50,
                "importance_threshold": 0.001,
                "correlation_threshold": 0.95
            },
            
            # Model interpretation
            "explainability": {
                "shap": True,
                "lime": True,
                "permutation_importance": True,
                "partial_dependence": True
            }
        }
        
        # Apply ML mode configuration
        self._apply_ml_mode_config()
        
        # Data sources
        self.DATA_SOURCES = {
            "yahoo_finance": True,
            "mock_data": True,  # Use mock data if real data unavailable
            "csv_files": True
        }
        
        # Logging configuration
        self.LOG_LEVEL = "INFO"
        self.LOG_FILE = os.path.join(self.PROJECT_ROOT, "logs", "fbx_hedging.log")
    
    def _apply_ml_mode_config(self):
        """Apply ML mode-specific configuration"""
        if self.ML_MODE == "minimal":
            self.ML_CONFIG = self._get_minimal_ml_config()
        elif self.ML_MODE == "production":
            self.ML_CONFIG = self._get_production_ml_config()
        elif self.ML_MODE == "testing":
            self.ML_CONFIG = self._get_testing_ml_config()
        # For "custom" mode, keep existing configuration
    
    def _get_minimal_ml_config(self):
        """Get minimal ML configuration for quick testing"""
        return {
            "feature_engineering": {
                "technical_indicators": True,
                "rolling_statistics": True,
                "lag_features": True,
                "interaction_features": False,
                "fourier_features": False,
                "wavelet_features": False,
                "pca_features": False,
                "polynomial_features": False
            },
            "models": {
                "random_forest": True,
                "ridge": True,
                "xgboost": False,
                "lstm": False,
                "transformer": False,
                "svm": False,
                "elastic_net": False,
                "lasso": False,
                "gradient_boosting": False,
                "extra_trees": False,
                "neural_network": False
            },
            "ensemble": {
                "voting": True,
                "stacking": False,
                "blending": False,
                "bayesian_optimization": False
            },
            "training": {
                "train_split": 0.8,
                "validation_split": 0.1,
                "test_split": 0.1,
                "cross_validation_folds": 3,
                "time_series_split": True,
                "walk_forward_validation": False,
                "lookback_window": 60,
                "prediction_horizon": [1, 5],
                "max_iterations": 100,
                "early_stopping": True,
                "patience": 10
            },
            "hyperopt": {
                "method": "optuna",
                "n_trials": 20,
                "parallel": False,
                "pruning": True,
                "timeout": 600
            },
            "feature_selection": {
                "method": "recursive_feature_elimination",
                "max_features": 20,
                "importance_threshold": 0.01,
                "correlation_threshold": 0.9
            },
            "explainability": {
                "shap": False,
                "lime": False,
                "permutation_importance": True,
                "partial_dependence": False
            }
        }
    
    def _get_production_ml_config(self):
        """Get production ML configuration for full deployment"""
        return {
            "feature_engineering": {
                "technical_indicators": True,
                "rolling_statistics": True,
                "lag_features": True,
                "interaction_features": True,
                "fourier_features": True,
                "wavelet_features": True,
                "pca_features": True,
                "polynomial_features": True
            },
            "models": {
                "random_forest": True,
                "xgboost": True,
                "lstm": True,
                "transformer": False,
                "svm": True,
                "elastic_net": True,
                "ridge": True,
                "lasso": True,
                "gradient_boosting": True,
                "extra_trees": True,
                "neural_network": True
            },
            "ensemble": {
                "voting": True,
                "stacking": True,
                "blending": True,
                "bayesian_optimization": True
            },
            "training": {
                "train_split": 0.7,
                "validation_split": 0.15,
                "test_split": 0.15,
                "cross_validation_folds": 10,
                "time_series_split": True,
                "walk_forward_validation": True,
                "lookback_window": 252,
                "prediction_horizon": [1, 5, 10, 22, 63],
                "max_iterations": 2000,
                "early_stopping": True,
                "patience": 100
            },
            "hyperopt": {
                "method": "optuna",
                "n_trials": 500,
                "parallel": True,
                "pruning": True,
                "timeout": 7200
            },
            "feature_selection": {
                "method": "recursive_feature_elimination",
                "max_features": 100,
                "importance_threshold": 0.001,
                "correlation_threshold": 0.95
            },
            "explainability": {
                "shap": True,
                "lime": True,
                "permutation_importance": True,
                "partial_dependence": True
            }
        }
    
    def _get_testing_ml_config(self):
        """Get testing ML configuration for development"""
        return {
            "feature_engineering": {
                "technical_indicators": True,
                "rolling_statistics": True,
                "lag_features": True,
                "interaction_features": True,
                "fourier_features": False,
                "wavelet_features": False,
                "pca_features": True,
                "polynomial_features": False
            },
            "models": {
                "random_forest": True,
                "xgboost": True,
                "ridge": True,
                "lasso": True,
                "lstm": False,
                "transformer": False,
                "svm": False,
                "elastic_net": True,
                "gradient_boosting": True,
                "extra_trees": True,
                "neural_network": False
            },
            "ensemble": {
                "voting": True,
                "stacking": True,
                "blending": True,
                "bayesian_optimization": False
            },
            "training": {
                "train_split": 0.8,
                "validation_split": 0.1,
                "test_split": 0.1,
                "cross_validation_folds": 5,
                "time_series_split": True,
                "walk_forward_validation": True,
                "lookback_window": 120,
                "prediction_horizon": [1, 5, 10, 22],
                "max_iterations": 500,
                "early_stopping": True,
                "patience": 50
            },
            "hyperopt": {
                "method": "optuna",
                "n_trials": 100,
                "parallel": True,
                "pruning": True,
                "timeout": 1800
            },
            "feature_selection": {
                "method": "recursive_feature_elimination",
                "max_features": 50,
                "importance_threshold": 0.001,
                "correlation_threshold": 0.95
            },
            "explainability": {
                "shap": True,
                "lime": False,
                "permutation_importance": True,
                "partial_dependence": True
            }
        }
        
    def get_data_path(self, filename: str) -> str:
        """Get full path for data file"""
        return os.path.join(self.DATA_DIR, filename)
    
    def get_report_path(self, filename: str = None) -> str:
        """Get full path for report file"""
        if filename is None:
            filename = self.REPORT_NAME
        return os.path.join(self.REPORTS_DIR, filename)
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        if self.START_DATE >= self.END_DATE:
            raise ValueError("Start date must be before end date")
        
        if self.INITIAL_CAPITAL <= 0:
            raise ValueError("Initial capital must be positive")
        
        if not (0 <= self.TRANSACTION_COST <= 1):
            raise ValueError("Transaction cost must be between 0 and 1")
        
        if not (0 <= self.MAX_POSITION_SIZE <= 1):
            raise ValueError("Max position size must be between 0 and 1")
        
        return True

# Create a global config instance
config = Config()