"""
Advanced Ensemble Methods for Combining ML Models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class EnsembleManager:
    """Advanced ensemble methods for combining multiple ML models"""
    
    def __init__(self, config):
        self.config = config
        self.ml_config = config.ML_CONFIG
        self.logger = logging.getLogger(__name__)
        self.ensemble_models = {}
        self.blending_weights = {}
        self.stacking_model = None
        
    def create_voting_ensemble(self, models: Dict[str, Any], X_val: pd.DataFrame, 
                             y_val: pd.Series) -> Any:
        """
        Create voting ensemble from multiple models
        
        Args:
            models: Dictionary of trained models
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Voting ensemble model
        """
        self.logger.info("Creating voting ensemble...")
        
        # Prepare estimators for voting
        estimators = []
        for name, model in models.items():
            if name != 'lstm':  # Skip LSTM for voting ensemble
                estimators.append((name, model))
        
        if len(estimators) < 2:
            self.logger.warning("Need at least 2 models for voting ensemble")
            return None
        
        # Create voting regressor
        voting_ensemble = VotingRegressor(
            estimators=estimators,
            n_jobs=-1
        )
        
        # Note: VotingRegressor doesn't need separate fitting if models are already trained
        # But we create a simple wrapper for consistency
        class VotingWrapper:
            def __init__(self, estimators):
                self.estimators = estimators
            
            def predict(self, X):
                predictions = []
                for name, model in self.estimators:
                    try:
                        pred = model.predict(X)
                        predictions.append(pred)
                    except:
                        continue
                
                if predictions:
                    return np.mean(predictions, axis=0)
                else:
                    return np.zeros(len(X))
        
        voting_wrapper = VotingWrapper(estimators)
        
        # Evaluate ensemble
        y_pred = voting_wrapper.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        
        self.logger.info(f"Voting ensemble R²: {r2:.4f}")
        self.ensemble_models['voting'] = voting_wrapper
        
        return voting_wrapper
    
    def create_stacking_ensemble(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                               y_train: pd.Series, X_val: pd.DataFrame, 
                               y_val: pd.Series) -> Any:
        """
        Create stacking ensemble using cross-validation
        
        Args:
            models: Dictionary of trained models
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Stacking ensemble model
        """
        self.logger.info("Creating stacking ensemble...")
        
        # Generate out-of-fold predictions for training data
        cv_predictions = self._generate_cv_predictions(models, X_train, y_train)
        
        # Train meta-model
        meta_model = LinearRegression()
        meta_model.fit(cv_predictions, y_train)
        
        # Generate predictions for validation data
        val_predictions = self._generate_base_predictions(models, X_val)
        
        # Create stacking wrapper
        class StackingWrapper:
            def __init__(self, base_models, meta_model):
                self.base_models = base_models
                self.meta_model = meta_model
            
            def predict(self, X):
                # Get base model predictions
                base_predictions = []
                for name, model in self.base_models.items():
                    try:
                        pred = model.predict(X)
                        base_predictions.append(pred)
                    except:
                        continue
                
                if base_predictions:
                    base_predictions = np.column_stack(base_predictions)
                    return self.meta_model.predict(base_predictions)
                else:
                    return np.zeros(len(X))
        
        stacking_wrapper = StackingWrapper(models, meta_model)
        
        # Evaluate ensemble
        y_pred = stacking_wrapper.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        
        self.logger.info(f"Stacking ensemble R²: {r2:.4f}")
        self.ensemble_models['stacking'] = stacking_wrapper
        self.stacking_model = meta_model
        
        return stacking_wrapper
    
    def create_blending_ensemble(self, models: Dict[str, Any], X_val: pd.DataFrame, 
                               y_val: pd.Series) -> Any:
        """
        Create blending ensemble with optimized weights
        
        Args:
            models: Dictionary of trained models
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Blending ensemble model
        """
        self.logger.info("Creating blending ensemble...")
        
        # Generate predictions for all models
        predictions = {}
        for name, model in models.items():
            try:
                pred = model.predict(X_val)
                predictions[name] = pred
            except Exception as e:
                self.logger.warning(f"Error getting predictions from {name}: {e}")
                continue
        
        if len(predictions) < 2:
            self.logger.warning("Need at least 2 models for blending ensemble")
            return None
        
        # Optimize weights using least squares
        pred_matrix = np.column_stack(list(predictions.values()))
        
        # Solve for optimal weights
        weights = np.linalg.lstsq(pred_matrix, y_val, rcond=None)[0]
        
        # Ensure weights are non-negative and sum to 1
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        # Store weights
        self.blending_weights = dict(zip(predictions.keys(), weights))
        
        # Create blending wrapper
        class BlendingWrapper:
            def __init__(self, base_models, weights):
                self.base_models = base_models
                self.weights = weights
            
            def predict(self, X):
                predictions = []
                model_names = []
                
                for name, model in self.base_models.items():
                    try:
                        pred = model.predict(X)
                        predictions.append(pred)
                        model_names.append(name)
                    except:
                        continue
                
                if predictions:
                    pred_matrix = np.column_stack(predictions)
                    # Use corresponding weights
                    used_weights = np.array([self.weights.get(name, 0) for name in model_names])
                    used_weights = used_weights / used_weights.sum()  # Renormalize
                    
                    return np.dot(pred_matrix, used_weights)
                else:
                    return np.zeros(len(X))
        
        blending_wrapper = BlendingWrapper(models, self.blending_weights)
        
        # Evaluate ensemble
        y_pred = blending_wrapper.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        
        self.logger.info(f"Blending ensemble R²: {r2:.4f}")
        self.logger.info(f"Blending weights: {self.blending_weights}")
        
        self.ensemble_models['blending'] = blending_wrapper
        
        return blending_wrapper
    
    def create_dynamic_ensemble(self, models: Dict[str, Any], X_val: pd.DataFrame, 
                              y_val: pd.Series) -> Any:
        """
        Create dynamic ensemble that adapts weights based on recent performance
        
        Args:
            models: Dictionary of trained models
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dynamic ensemble model
        """
        self.logger.info("Creating dynamic ensemble...")
        
        # Calculate rolling performance for each model
        window_size = min(50, len(X_val) // 4)  # Use 25% of data or 50 points
        
        model_performances = {}
        for name, model in models.items():
            try:
                pred = model.predict(X_val)
                
                # Calculate rolling R²
                rolling_r2 = []
                for i in range(window_size, len(y_val)):
                    window_true = y_val.iloc[i-window_size:i]
                    window_pred = pred[i-window_size:i]
                    
                    if len(window_true) > 1:
                        r2 = r2_score(window_true, window_pred)
                        rolling_r2.append(max(r2, 0))  # Ensure non-negative
                    else:
                        rolling_r2.append(0)
                
                model_performances[name] = rolling_r2
                
            except Exception as e:
                self.logger.warning(f"Error calculating performance for {name}: {e}")
                continue
        
        if not model_performances:
            self.logger.warning("No valid models for dynamic ensemble")
            return None
        
        # Create dynamic wrapper
        class DynamicWrapper:
            def __init__(self, base_models, performances, window_size):
                self.base_models = base_models
                self.performances = performances
                self.window_size = window_size
            
            def predict(self, X):
                # For simplicity, use average weights from historical performance
                # In practice, this would be more sophisticated
                avg_performances = {}
                for name, perf_list in self.performances.items():
                    avg_performances[name] = np.mean(perf_list) if perf_list else 0
                
                # Normalize to get weights
                total_perf = sum(avg_performances.values())
                if total_perf > 0:
                    weights = {name: perf / total_perf for name, perf in avg_performances.items()}
                else:
                    weights = {name: 1/len(avg_performances) for name in avg_performances.keys()}
                
                # Generate weighted predictions
                predictions = []
                model_names = []
                
                for name, model in self.base_models.items():
                    try:
                        pred = model.predict(X)
                        predictions.append(pred)
                        model_names.append(name)
                    except:
                        continue
                
                if predictions:
                    pred_matrix = np.column_stack(predictions)
                    used_weights = np.array([weights.get(name, 0) for name in model_names])
                    used_weights = used_weights / used_weights.sum()
                    
                    return np.dot(pred_matrix, used_weights)
                else:
                    return np.zeros(len(X))
        
        dynamic_wrapper = DynamicWrapper(models, model_performances, window_size)
        
        # Evaluate ensemble
        y_pred = dynamic_wrapper.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        
        self.logger.info(f"Dynamic ensemble R²: {r2:.4f}")
        self.ensemble_models['dynamic'] = dynamic_wrapper
        
        return dynamic_wrapper
    
    def _generate_cv_predictions(self, models: Dict[str, Any], X: pd.DataFrame, 
                               y: pd.Series) -> pd.DataFrame:
        """Generate cross-validation predictions for stacking"""
        cv_predictions = pd.DataFrame(index=X.index)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for name, model in models.items():
            if name == 'lstm':  # Skip LSTM for simplicity
                continue
            
            fold_predictions = np.zeros(len(X))
            
            try:
                for train_idx, val_idx in tscv.split(X):
                    X_fold_train = X.iloc[train_idx]
                    y_fold_train = y.iloc[train_idx]
                    X_fold_val = X.iloc[val_idx]
                    
                    # Clone and train model
                    from sklearn.base import clone
                    fold_model = clone(model)
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Predict on validation fold
                    fold_predictions[val_idx] = fold_model.predict(X_fold_val)
                
                cv_predictions[name] = fold_predictions
                
            except Exception as e:
                self.logger.warning(f"Error in CV predictions for {name}: {e}")
                continue
        
        return cv_predictions
    
    def _generate_base_predictions(self, models: Dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions from base models"""
        predictions = pd.DataFrame(index=X.index)
        
        for name, model in models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                self.logger.warning(f"Error getting predictions from {name}: {e}")
                continue
        
        return predictions
    
    def evaluate_ensembles(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all ensemble models"""
        self.logger.info("Evaluating ensemble models...")
        
        results = {}
        
        for name, ensemble in self.ensemble_models.items():
            try:
                y_pred = ensemble.predict(X_test)
                
                # Calculate metrics
                results[name] = {
                    'r2_score': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
                
                self.logger.info(f"{name} ensemble - R²: {results[name]['r2_score']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {name} ensemble: {e}")
                continue
        
        return results
    
    def get_best_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[str, Any]:
        """Get the best performing ensemble model"""
        evaluation_results = self.evaluate_ensembles(X_test, y_test)
        
        if not evaluation_results:
            return None, None
        
        # Find best ensemble based on R²
        best_name = max(evaluation_results.keys(), 
                       key=lambda x: evaluation_results[x]['r2_score'])
        
        return best_name, self.ensemble_models[best_name]
    
    def save_ensembles(self, save_dir: str) -> None:
        """Save ensemble models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, ensemble in self.ensemble_models.items():
            try:
                ensemble_path = os.path.join(save_dir, f"ensemble_{name}.pkl")
                joblib.dump(ensemble, ensemble_path)
                self.logger.info(f"Saved {name} ensemble to {ensemble_path}")
            except Exception as e:
                self.logger.error(f"Error saving {name} ensemble: {e}")
        
        # Save blending weights
        if self.blending_weights:
            weights_path = os.path.join(save_dir, "blending_weights.pkl")
            joblib.dump(self.blending_weights, weights_path)
    
    def load_ensembles(self, load_dir: str) -> None:
        """Load ensemble models"""
        import os
        
        for file_name in os.listdir(load_dir):
            if file_name.startswith("ensemble_") and file_name.endswith(".pkl"):
                ensemble_name = file_name.replace("ensemble_", "").replace(".pkl", "")
                ensemble_path = os.path.join(load_dir, file_name)
                
                try:
                    self.ensemble_models[ensemble_name] = joblib.load(ensemble_path)
                    self.logger.info(f"Loaded {ensemble_name} ensemble from {ensemble_path}")
                except Exception as e:
                    self.logger.error(f"Error loading {ensemble_name} ensemble: {e}")
        
        # Load blending weights
        weights_path = os.path.join(load_dir, "blending_weights.pkl")
        if os.path.exists(weights_path):
            self.blending_weights = joblib.load(weights_path)
