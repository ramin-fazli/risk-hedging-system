"""
ML Configuration Presets for Different Use Cases
"""

from datetime import datetime, timedelta

class MLConfigPresets:
    """Predefined ML configurations for different scenarios"""
    
    @staticmethod
    def get_quick_test_config():
        """Fast configuration for testing ML functionality"""
        return {
            'feature_engineering': {
                'technical_indicators': True,
                'rolling_statistics': True,
                'lag_features': True,
                'interaction_features': False,  # Disable for speed
                'fourier_features': False,
                'wavelet_features': False,
                'pca_features': True,
                'polynomial_features': False
            },
            'models': {
                'random_forest': True,
                'ridge': True,
                'xgboost': False,  # Disable heavy models
                'lstm': False,
                'svm': False,
                'elastic_net': False,
                'gradient_boosting': False,
                'extra_trees': False,
                'neural_network': False,
                'lasso': True
            },
            'ensemble': {
                'voting': True,
                'stacking': False,
                'blending': False,
                'bayesian_optimization': False
            },
            'training': {
                'train_split': 0.8,
                'validation_split': 0.1,
                'test_split': 0.1,
                'cross_validation_folds': 3,  # Reduced for speed
                'time_series_split': True,
                'walk_forward_validation': False,
                'lookback_window': 126,  # 6 months
                'prediction_horizon': [1, 5],  # Just 1d and 1w
                'max_iterations': 100,
                'early_stopping': True,
                'patience': 10
            },
            'hyperopt': {
                'method': 'optuna',
                'n_trials': 5,  # Very limited for testing
                'parallel': False,
                'pruning': True,
                'timeout': 300  # 5 minutes
            },
            'feature_selection': {
                'method': 'recursive_feature_elimination',
                'max_features': 20,  # Reduced for speed
                'importance_threshold': 0.01,
                'correlation_threshold': 0.9
            },
            'explainability': {
                'shap': False,  # Disable for speed
                'lime': False,
                'permutation_importance': True,
                'partial_dependence': False
            }
        }
    
    @staticmethod
    def get_production_config():
        """Comprehensive configuration for production use"""
        return {
            'feature_engineering': {
                'technical_indicators': True,
                'rolling_statistics': True,
                'lag_features': True,
                'interaction_features': True,
                'fourier_features': True,
                'wavelet_features': False,  # Optional - computationally expensive
                'pca_features': True,
                'polynomial_features': True
            },
            'models': {
                'random_forest': True,
                'xgboost': True,
                'lstm': True,
                'transformer': False,  # Very expensive
                'svm': True,
                'elastic_net': True,
                'ridge': True,
                'lasso': True,
                'gradient_boosting': True,
                'extra_trees': True,
                'neural_network': True
            },
            'ensemble': {
                'voting': True,
                'stacking': True,
                'blending': True,
                'bayesian_optimization': False
            },
            'training': {
                'train_split': 0.7,
                'validation_split': 0.15,
                'test_split': 0.15,
                'cross_validation_folds': 5,
                'time_series_split': True,
                'walk_forward_validation': True,
                'lookback_window': 252,
                'prediction_horizon': [1, 5, 10, 22],
                'max_iterations': 1000,
                'early_stopping': True,
                'patience': 50
            },
            'hyperopt': {
                'method': 'optuna',
                'n_trials': 100,
                'parallel': True,
                'pruning': True,
                'timeout': 3600  # 1 hour
            },
            'feature_selection': {
                'method': 'recursive_feature_elimination',
                'max_features': 100,
                'importance_threshold': 0.001,
                'correlation_threshold': 0.95
            },
            'explainability': {
                'shap': True,
                'lime': True,
                'permutation_importance': True,
                'partial_dependence': True
            }
        }
    
    @staticmethod
    def get_research_config():
        """Configuration for research and experimentation"""
        return {
            'feature_engineering': {
                'technical_indicators': True,
                'rolling_statistics': True,
                'lag_features': True,
                'interaction_features': True,
                'fourier_features': True,
                'wavelet_features': True,  # Enable for research
                'pca_features': True,
                'polynomial_features': True
            },
            'models': {
                'random_forest': True,
                'xgboost': True,
                'lstm': True,
                'transformer': True,  # Enable for research
                'svm': True,
                'elastic_net': True,
                'ridge': True,
                'lasso': True,
                'gradient_boosting': True,
                'extra_trees': True,
                'neural_network': True
            },
            'ensemble': {
                'voting': True,
                'stacking': True,
                'blending': True,
                'bayesian_optimization': True
            },
            'training': {
                'train_split': 0.6,
                'validation_split': 0.2,
                'test_split': 0.2,
                'cross_validation_folds': 10,
                'time_series_split': True,
                'walk_forward_validation': True,
                'lookback_window': 504,  # 2 years
                'prediction_horizon': [1, 3, 5, 10, 22, 66],  # Multiple horizons
                'max_iterations': 2000,
                'early_stopping': True,
                'patience': 100
            },
            'hyperopt': {
                'method': 'optuna',
                'n_trials': 500,  # Extensive search
                'parallel': True,
                'pruning': True,
                'timeout': 7200  # 2 hours
            },
            'feature_selection': {
                'method': 'recursive_feature_elimination',
                'max_features': 200,
                'importance_threshold': 0.0001,
                'correlation_threshold': 0.98
            },
            'explainability': {
                'shap': True,
                'lime': True,
                'permutation_importance': True,
                'partial_dependence': True
            }
        }
    
    @staticmethod
    def get_lightweight_config():
        """Minimal configuration for resource-constrained environments"""
        return {
            'feature_engineering': {
                'technical_indicators': False,  # Disable TA-Lib features
                'rolling_statistics': True,
                'lag_features': True,
                'interaction_features': False,
                'fourier_features': False,
                'wavelet_features': False,
                'pca_features': False,
                'polynomial_features': False
            },
            'models': {
                'random_forest': False,
                'xgboost': False,
                'lstm': False,
                'transformer': False,
                'svm': False,
                'elastic_net': True,
                'ridge': True,
                'lasso': True,
                'gradient_boosting': False,
                'extra_trees': False,
                'neural_network': False
            },
            'ensemble': {
                'voting': False,
                'stacking': False,
                'blending': False,
                'bayesian_optimization': False
            },
            'training': {
                'train_split': 0.8,
                'validation_split': 0.1,
                'test_split': 0.1,
                'cross_validation_folds': 3,
                'time_series_split': True,
                'walk_forward_validation': False,
                'lookback_window': 63,  # 3 months
                'prediction_horizon': [1],  # Just 1 day
                'max_iterations': 100,
                'early_stopping': True,
                'patience': 10
            },
            'hyperopt': {
                'method': 'grid_search',  # Simple grid search
                'n_trials': 10,
                'parallel': False,
                'pruning': False,
                'timeout': 300
            },
            'feature_selection': {
                'method': 'recursive_feature_elimination',
                'max_features': 10,
                'importance_threshold': 0.1,
                'correlation_threshold': 0.8
            },
            'explainability': {
                'shap': False,
                'lime': False,
                'permutation_importance': False,
                'partial_dependence': False
            }
        }
    
    @staticmethod
    def get_demo_config():
        """Configuration for demonstrations and tutorials"""
        return {
            'feature_engineering': {
                'technical_indicators': True,
                'rolling_statistics': True,
                'lag_features': True,
                'interaction_features': True,
                'fourier_features': True,
                'wavelet_features': False,
                'pca_features': True,
                'polynomial_features': False
            },
            'models': {
                'random_forest': True,
                'xgboost': True,
                'lstm': False,  # Skip for demo speed
                'transformer': False,
                'svm': False,
                'elastic_net': True,
                'ridge': True,
                'lasso': False,
                'gradient_boosting': True,
                'extra_trees': False,
                'neural_network': False
            },
            'ensemble': {
                'voting': True,
                'stacking': True,
                'blending': False,
                'bayesian_optimization': False
            },
            'training': {
                'train_split': 0.7,
                'validation_split': 0.15,
                'test_split': 0.15,
                'cross_validation_folds': 3,
                'time_series_split': True,
                'walk_forward_validation': False,
                'lookback_window': 252,
                'prediction_horizon': [1, 5, 22],
                'max_iterations': 200,
                'early_stopping': True,
                'patience': 20
            },
            'hyperopt': {
                'method': 'optuna',
                'n_trials': 20,
                'parallel': True,
                'pruning': True,
                'timeout': 600  # 10 minutes
            },
            'feature_selection': {
                'method': 'recursive_feature_elimination',
                'max_features': 50,
                'importance_threshold': 0.005,
                'correlation_threshold': 0.9
            },
            'explainability': {
                'shap': True,
                'lime': False,
                'permutation_importance': True,
                'partial_dependence': True
            }
        }
    
    @staticmethod
    def apply_preset(config, preset_name):
        """Apply a preset to the existing configuration"""
        presets = {
            'quick_test': MLConfigPresets.get_quick_test_config(),
            'production': MLConfigPresets.get_production_config(),
            'research': MLConfigPresets.get_research_config(),
            'lightweight': MLConfigPresets.get_lightweight_config(),
            'demo': MLConfigPresets.get_demo_config()
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        # Update ML configuration
        config.ML_CONFIG.update(presets[preset_name])
        
        return config
