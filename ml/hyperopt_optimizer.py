"""
Advanced Hyperparameter Optimization using Optuna
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Optional import for Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class HyperoptOptimizer:
    """Advanced hyperparameter optimization using Optuna"""
    
    def __init__(self, config):
        self.config = config
        self.ml_config = config.ML_CONFIG
        self.logger = logging.getLogger(__name__)
        self.study = None
        self.best_params = {}
        self.optimization_history = {}
        
    def optimize_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model
        
        Args:
            model_name: Name of the model to optimize
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with best parameters and optimization results
        """
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, using default parameters")
            return self._get_default_params(model_name)
        
        self.logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Define objective function
        def objective(trial):
            return self._objective_function(
                trial, model_name, X_train, y_train, X_val, y_val
            )
        
        # Run optimization
        try:
            study.optimize(
                objective,
                n_trials=self.ml_config['hyperopt']['n_trials'],
                timeout=self.ml_config['hyperopt']['timeout'],
                show_progress_bar=True
            )
            
            # Store results
            self.best_params[model_name] = study.best_params
            self.optimization_history[model_name] = {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
                'optimization_history': [(t.number, t.value) for t in study.trials if t.value is not None]
            }
            
            self.logger.info(f"Best parameters for {model_name}: {study.best_params}")
            self.logger.info(f"Best score: {study.best_value:.4f}")
            
            return {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'study': study
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing {model_name}: {e}")
            return self._get_default_params(model_name)
    
    def _objective_function(self, trial, model_name: str, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Objective function for hyperparameter optimization"""
        
        # Get model with trial parameters
        model = self._get_model_with_params(model_name, trial)
        
        if model is None:
            return -np.inf
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate score (R²)
            score = r2_score(y_val, y_pred)
            
            # Handle NaN or inf values
            if np.isnan(score) or np.isinf(score):
                return -np.inf
            
            return score
            
        except Exception as e:
            self.logger.debug(f"Trial failed for {model_name}: {e}")
            return -np.inf
    
    def _get_model_with_params(self, model_name: str, trial) -> Optional[Any]:
        """Get model instance with trial parameters"""
        
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                random_state=42,
                n_jobs=-1
            )
        
        elif model_name == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
                    reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
                    random_state=42,
                    n_jobs=-1
                )
            except ImportError:
                return None
        
        elif model_name == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                random_state=42
            )
        
        elif model_name == 'extra_trees':
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                random_state=42,
                n_jobs=-1
            )
        
        elif model_name == 'ridge':
            from sklearn.linear_model import Ridge
            return Ridge(
                alpha=trial.suggest_float('alpha', 0.001, 100, log=True),
                solver=trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
                random_state=42
            )
        
        elif model_name == 'lasso':
            from sklearn.linear_model import Lasso
            return Lasso(
                alpha=trial.suggest_float('alpha', 0.001, 10, log=True),
                max_iter=trial.suggest_int('max_iter', 1000, 5000),
                random_state=42
            )
        
        elif model_name == 'elastic_net':
            from sklearn.linear_model import ElasticNet
            return ElasticNet(
                alpha=trial.suggest_float('alpha', 0.001, 10, log=True),
                l1_ratio=trial.suggest_float('l1_ratio', 0.1, 0.9),
                max_iter=trial.suggest_int('max_iter', 1000, 5000),
                random_state=42
            )
        
        elif model_name == 'svm':
            from sklearn.svm import SVR
            return SVR(
                C=trial.suggest_float('C', 0.1, 100, log=True),
                epsilon=trial.suggest_float('epsilon', 0.01, 1.0),
                kernel=trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                gamma=trial.suggest_categorical('gamma', ['scale', 'auto']) if 
                      trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']) in ['rbf', 'poly'] else 'scale'
            )
        
        elif model_name == 'neural_network':
            from sklearn.neural_network import MLPRegressor
            
            # Suggest number of hidden layers
            n_layers = trial.suggest_int('n_layers', 1, 3)
            
            # Suggest hidden layer sizes
            hidden_layer_sizes = []
            for i in range(n_layers):
                size = trial.suggest_int(f'layer_{i}_size', 10, 200)
                hidden_layer_sizes.append(size)
            
            return MLPRegressor(
                hidden_layer_sizes=tuple(hidden_layer_sizes),
                activation=trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                solver=trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                alpha=trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                learning_rate=trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                max_iter=trial.suggest_int('max_iter', 200, 1000),
                random_state=42
            )
        
        else:
            self.logger.warning(f"Unknown model name: {model_name}")
            return None
    
    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters for a model"""
        default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt'
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            },
            'extra_trees': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 0.1},
            'elastic_net': {'alpha': 0.1, 'l1_ratio': 0.5},
            'svm': {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'},
            'neural_network': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001
            }
        }
        
        return {
            'best_params': default_params.get(model_name, {}),
            'best_score': 0.0,
            'study': None
        }
    
    def optimize_all_models(self, model_names: List[str], X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Optimize hyperparameters for all specified models"""
        self.logger.info("Starting hyperparameter optimization for all models...")
        
        optimization_results = {}
        
        for model_name in model_names:
            if model_name == 'lstm':  # Skip LSTM due to complexity
                continue
            
            try:
                result = self.optimize_model(model_name, X_train, y_train, X_val, y_val)
                optimization_results[model_name] = result
                
            except Exception as e:
                self.logger.error(f"Error optimizing {model_name}: {e}")
                optimization_results[model_name] = self._get_default_params(model_name)
        
        return optimization_results
    
    def get_optimized_model(self, model_name: str) -> Optional[Any]:
        """Get model instance with optimized parameters"""
        if model_name not in self.best_params:
            self.logger.warning(f"No optimized parameters found for {model_name}")
            return None
        
        params = self.best_params[model_name]
        
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        
        elif model_name == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            except ImportError:
                return None
        
        elif model_name == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**params, random_state=42)
        
        elif model_name == 'extra_trees':
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
        
        elif model_name == 'ridge':
            from sklearn.linear_model import Ridge
            return Ridge(**params, random_state=42)
        
        elif model_name == 'lasso':
            from sklearn.linear_model import Lasso
            return Lasso(**params, random_state=42)
        
        elif model_name == 'elastic_net':
            from sklearn.linear_model import ElasticNet
            return ElasticNet(**params, random_state=42)
        
        elif model_name == 'svm':
            from sklearn.svm import SVR
            return SVR(**params)
        
        elif model_name == 'neural_network':
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(**params, random_state=42)
        
        else:
            self.logger.warning(f"Unknown model name: {model_name}")
            return None
    
    def save_optimization_results(self, save_dir: str) -> None:
        """Save optimization results to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best parameters
        if self.best_params:
            params_path = os.path.join(save_dir, "best_params.pkl")
            joblib.dump(self.best_params, params_path)
            self.logger.info(f"Saved best parameters to {params_path}")
        
        # Save optimization history
        if self.optimization_history:
            history_path = os.path.join(save_dir, "optimization_history.pkl")
            joblib.dump(self.optimization_history, history_path)
            self.logger.info(f"Saved optimization history to {history_path}")
    
    def load_optimization_results(self, load_dir: str) -> None:
        """Load optimization results from disk"""
        # Load best parameters
        params_path = os.path.join(load_dir, "best_params.pkl")
        if os.path.exists(params_path):
            self.best_params = joblib.load(params_path)
            self.logger.info(f"Loaded best parameters from {params_path}")
        
        # Load optimization history
        history_path = os.path.join(load_dir, "optimization_history.pkl")
        if os.path.exists(history_path):
            self.optimization_history = joblib.load(history_path)
            self.logger.info(f"Loaded optimization history from {history_path}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        summary = {
            'optimized_models': list(self.best_params.keys()),
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }
        
        # Add best scores
        best_scores = {}
        for model_name, history in self.optimization_history.items():
            best_scores[model_name] = history.get('best_value', 0.0)
        
        summary['best_scores'] = best_scores
        
        return summary
