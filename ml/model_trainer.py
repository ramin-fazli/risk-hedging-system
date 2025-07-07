"""
Advanced Model Training with Multiple ML Algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Optional imports for advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class ModelTrainer:
    """Advanced model training with multiple ML algorithms"""
    
    def __init__(self, config):
        self.config = config
        self.ml_config = config.ML_CONFIG
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.model_performances = {}
        self.best_model = None
        self.best_score = -np.inf
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train multiple ML models and return performance metrics
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of trained models and their performances
        """
        self.logger.info("Starting model training...")
        
        # Initialize models
        models_to_train = self._get_models_to_train()
        
        # Train each model
        for model_name, model in models_to_train.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Special handling for LSTM
                if model_name == 'lstm':
                    trained_model = self._train_lstm(X_train, y_train, X_val, y_val)
                else:
                    # Standard sklearn-like models
                    trained_model = model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = trained_model.predict(X_val)
                
                # Calculate metrics
                performance = self._calculate_metrics(y_val, y_pred)
                
                # Store model and performance
                self.models[model_name] = trained_model
                self.model_performances[model_name] = performance
                
                # Update best model
                if performance['r2_score'] > self.best_score:
                    self.best_score = performance['r2_score']
                    self.best_model = model_name
                
                self.logger.info(f"{model_name} - R²: {performance['r2_score']:.4f}, "
                               f"MSE: {performance['mse']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                continue
        
        self.logger.info(f"Best model: {self.best_model} with R² score: {self.best_score:.4f}")
        
        return {
            'models': self.models,
            'performances': self.model_performances,
            'best_model': self.best_model
        }
    
    def _get_models_to_train(self) -> Dict[str, Any]:
        """Get dictionary of models to train based on configuration"""
        models = {}
        
        # Random Forest
        if self.ml_config['models']['random_forest']:
            models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        # Extra Trees
        if self.ml_config['models']['extra_trees']:
            models['extra_trees'] = ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        # Gradient Boosting
        if self.ml_config['models']['gradient_boosting']:
            models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        # XGBoost
        if self.ml_config['models']['xgboost'] and XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        # Ridge Regression
        if self.ml_config['models']['ridge']:
            models['ridge'] = Ridge(
                alpha=1.0,
                random_state=42
            )
        
        # Lasso Regression
        if self.ml_config['models']['lasso']:
            models['lasso'] = Lasso(
                alpha=0.1,
                random_state=42,
                max_iter=1000
            )
        
        # Elastic Net
        if self.ml_config['models']['elastic_net']:
            models['elastic_net'] = ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42,
                max_iter=1000
            )
        
        # Support Vector Regression
        if self.ml_config['models']['svm']:
            models['svm'] = SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale'
            )
        
        # Neural Network (MLP)
        if self.ml_config['models']['neural_network']:
            models['neural_network'] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
        
        return models
    
    def _train_lstm(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """Train LSTM model for time series prediction"""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available, skipping LSTM")
            return None
        
        # Prepare data for LSTM (3D reshape)
        def prepare_lstm_data(X, y, lookback=60):
            X_lstm = []
            y_lstm = []
            
            for i in range(lookback, len(X)):
                X_lstm.append(X.iloc[i-lookback:i].values)
                y_lstm.append(y.iloc[i])
            
            return np.array(X_lstm), np.array(y_lstm)
        
        # Prepare training data
        X_train_lstm, y_train_lstm = prepare_lstm_data(X_train, y_train)
        X_val_lstm, y_val_lstm = prepare_lstm_data(X_val, y_val)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train_lstm, y_train_lstm,
            validation_data=(X_val_lstm, y_val_lstm),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Create wrapper for sklearn-like interface
        class LSTMWrapper:
            def __init__(self, model, lookback=60):
                self.model = model
                self.lookback = lookback
            
            def predict(self, X):
                X_lstm, _ = prepare_lstm_data(X, pd.Series(np.zeros(len(X))), self.lookback)
                return self.model.predict(X_lstm, verbose=0).flatten()
        
        return LSTMWrapper(model)
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        # Handle NaN values
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'mse': np.inf,
                'rmse': np.inf,
                'mae': np.inf,
                'r2_score': -np.inf
            }
        
        return {
            'mse': mean_squared_error(y_true_clean, y_pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'r2_score': r2_score(y_true_clean, y_pred_clean)
        }
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation for all models"""
        self.logger.info("Starting cross-validation...")
        
        cv_results = {}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.ml_config['training']['cross_validation_folds'])
        
        for model_name, model in self.models.items():
            if model_name == 'lstm':
                continue  # Skip LSTM for cross-validation due to complexity
            
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X, y, cv=tscv, scoring='r2', n_jobs=-1
                )
                
                cv_results[model_name] = {
                    'mean_r2': cv_scores.mean(),
                    'std_r2': cv_scores.std(),
                    'scores': cv_scores.tolist()
                }
                
                self.logger.info(f"{model_name} CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error in cross-validation for {model_name}: {e}")
        
        return cv_results
    
    def get_feature_importance(self, model_name: str) -> Optional[pd.Series]:
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        # Tree-based models
        if hasattr(model, 'feature_importances_'):
            return pd.Series(model.feature_importances_, index=model.feature_names_in_)
        
        # Linear models
        elif hasattr(model, 'coef_'):
            return pd.Series(np.abs(model.coef_), index=model.feature_names_in_)
        
        return None
    
    def save_models(self, save_dir: str) -> None:
        """Save trained models to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            
            try:
                if model_name == 'lstm':
                    # Save LSTM model separately
                    model.model.save(os.path.join(save_dir, f"{model_name}.h5"))
                else:
                    joblib.dump(model, model_path)
                
                self.logger.info(f"Saved {model_name} to {model_path}")
            
            except Exception as e:
                self.logger.error(f"Error saving {model_name}: {e}")
        
        # Save performance metrics
        performance_path = os.path.join(save_dir, "model_performances.pkl")
        joblib.dump(self.model_performances, performance_path)
    
    def load_models(self, load_dir: str) -> None:
        """Load trained models from disk"""
        self.models = {}
        
        for model_file in os.listdir(load_dir):
            if model_file.endswith('.pkl') and model_file != 'model_performances.pkl':
                model_name = model_file.replace('.pkl', '')
                model_path = os.path.join(load_dir, model_file)
                
                try:
                    self.models[model_name] = joblib.load(model_path)
                    self.logger.info(f"Loaded {model_name} from {model_path}")
                except Exception as e:
                    self.logger.error(f"Error loading {model_name}: {e}")
        
        # Load performance metrics
        performance_path = os.path.join(load_dir, "model_performances.pkl")
        if os.path.exists(performance_path):
            self.model_performances = joblib.load(performance_path)
    
    def predict_with_model(self, model_name: str, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Make predictions with a specific model"""
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            return None
        
        try:
            return self.models[model_name].predict(X)
        except Exception as e:
            self.logger.error(f"Error making predictions with {model_name}: {e}")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        summary = {
            'num_models': len(self.models),
            'models': list(self.models.keys()),
            'best_model': self.best_model,
            'best_score': self.best_score,
            'performances': self.model_performances
        }
        
        return summary
