"""
Simplified ML Configuration for Testing
"""

def get_minimal_ml_config():
    """Get minimal ML configuration for testing"""
    return {
        'feature_engineering': {
            'technical_indicators': True,
            'rolling_statistics': True,
            'lag_features': True,
            'interaction_features': False,  # Disable to reduce complexity
            'fourier_features': False,
            'wavelet_features': False,
            'pca_features': False,
            'polynomial_features': False
        },
        
        'models': {
            'random_forest': True,
            'xgboost': False,  # Disable heavy models
            'lstm': False,
            'transformer': False,
            'svm': False,
            'elastic_net': True,
            'ridge': True,
            'lasso': False,
            'gradient_boosting': False,
            'extra_trees': False,
            'neural_network': False
        },
        
        'ensemble': {
            'voting': True,
            'stacking': False,  # Disable for testing
            'blending': False,
            'bayesian_optimization': False
        },
        
        'training': {
            'train_split': 0.8,
            'validation_split': 0.1,
            'test_split': 0.1,
            'cross_validation_folds': 3,  # Reduce for speed
            'time_series_split': True,
            'walk_forward_validation': True,
            'lookback_window': 60,  # Reduce for speed
            'prediction_horizon': [1, 5],  # Reduce horizons
            'max_iterations': 100,
            'early_stopping': True,
            'patience': 10
        },
        
        'hyperopt': {
            'method': 'optuna',
            'n_trials': 20,  # Reduce for testing
            'parallel': True,
            'pruning': True,
            'timeout': 600  # 10 minutes
        },
        
        'feature_selection': {
            'method': 'recursive_feature_elimination',
            'max_features': 20,  # Reduce for testing
            'importance_threshold': 0.01,
            'correlation_threshold': 0.95
        },
        
        'explainability': {
            'shap': False,  # Disable for testing
            'lime': False,
            'permutation_importance': True,
            'partial_dependence': False
        }
    }

def get_production_ml_config():
    """Get full production ML configuration"""
    return {
        'feature_engineering': {
            'technical_indicators': True,
            'rolling_statistics': True,
            'lag_features': True,
            'interaction_features': True,
            'fourier_features': True,
            'wavelet_features': True,
            'pca_features': True,
            'polynomial_features': True
        },
        
        'models': {
            'random_forest': True,
            'xgboost': True,
            'lstm': True,
            'transformer': False,  # Very intensive
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
            'train_split': 0.8,
            'validation_split': 0.1,
            'test_split': 0.1,
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
            'timeout': 3600
        },
        
        'feature_selection': {
            'method': 'recursive_feature_elimination',
            'max_features': 50,
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
