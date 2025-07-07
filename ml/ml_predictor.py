"""
Main ML Predictor - Orchestrates all ML functionality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from datetime import datetime

from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .ensemble_manager import EnsembleManager
from .hyperopt_optimizer import HyperoptOptimizer

class MLPredictor:
    """Main ML predictor that orchestrates all ML functionality"""
    
    def __init__(self, config):
        self.config = config
        self.ml_config = config.ML_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(config)
        self.model_trainer = ModelTrainer(config)
        self.ensemble_manager = EnsembleManager(config)
        self.hyperopt_optimizer = HyperoptOptimizer(config)
        
        # ML results
        self.features = None
        self.target = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Models and results
        self.trained_models = {}
        self.ensemble_models = {}
        self.best_model = None
        self.predictions = {}
        self.model_performances = {}
        
        # Create directories
        self.ml_output_dir = os.path.join(config.PROJECT_ROOT, "ml_output")
        os.makedirs(self.ml_output_dir, exist_ok=True)
    
    def run_full_ml_pipeline(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run the complete ML pipeline
        
        Args:
            data: Dictionary containing 'fbx', 'instruments', 'market', 'revenue' data
            
        Returns:
            Dictionary with ML results and predictions
        """
        if not self.config.ML_ENABLED:
            self.logger.info("ML functionality is disabled")
            return {'ml_enabled': False}
        
        self.logger.info("Starting full ML pipeline...")
        
        results = {
            'ml_enabled': True,
            'pipeline_start_time': datetime.now(),
            'status': 'running'
        }
        
        try:
            # Step 1: Feature Engineering
            self.logger.info("Step 1: Feature Engineering")
            self.features = self.feature_engineer.engineer_features(data)
            results['features_created'] = len(self.features.columns)
            
            # Step 2: Prepare target variable (FBX returns)
            self.logger.info("Step 2: Preparing target variable")
            self.target = self._prepare_target(data)
            results['target_prepared'] = True
            
            # Step 3: Feature Selection
            self.logger.info("Step 3: Feature Selection")
            self.features = self.feature_engineer.select_features(self.features, self.target)
            results['features_selected'] = len(self.features.columns)
            
            # Step 4: Data Splitting
            self.logger.info("Step 4: Data Splitting")
            self._split_data()
            results['data_split'] = True
            
            # Step 5: Feature Scaling
            self.logger.info("Step 5: Feature Scaling")
            self._scale_features()
            results['features_scaled'] = True
            
            # Step 6: Hyperparameter Optimization (if enabled)
            if self.ml_config['hyperopt']['method'] == 'optuna':
                self.logger.info("Step 6: Hyperparameter Optimization")
                self._optimize_hyperparameters()
                results['hyperopt_completed'] = True
            
            # Step 7: Model Training
            self.logger.info("Step 7: Model Training")
            training_results = self._train_models()
            results['models_trained'] = training_results['models_trained']
            results['best_single_model'] = training_results['best_model']
            
            # Step 8: Ensemble Creation
            self.logger.info("Step 8: Ensemble Creation")
            ensemble_results = self._create_ensembles()
            results['ensembles_created'] = ensemble_results['ensembles_created']
            results['best_ensemble'] = ensemble_results['best_ensemble']
            
            # Step 9: Model Evaluation
            self.logger.info("Step 9: Model Evaluation")
            evaluation_results = self._evaluate_models()
            results['evaluation_completed'] = True
            results['model_performances'] = evaluation_results
            
            # Step 10: Generate Predictions
            self.logger.info("Step 10: Generating Predictions")
            prediction_results = self._generate_predictions()
            results['predictions_generated'] = True
            results['predictions'] = prediction_results
            
            # Step 11: Save Results
            self.logger.info("Step 11: Saving Results")
            self._save_ml_results()
            results['results_saved'] = True
            
            results['status'] = 'completed'
            results['pipeline_end_time'] = datetime.now()
            results['total_time'] = (results['pipeline_end_time'] - results['pipeline_start_time']).total_seconds()
            
            self.logger.info(f"ML pipeline completed successfully in {results['total_time']:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in ML pipeline: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def _prepare_target(self, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Prepare target variable for prediction"""
        # Use FBX returns as target
        if 'fbx' in data and 'Returns' in data['fbx'].columns:
            target = data['fbx']['Returns'].dropna()
        else:
            # Fallback to FBX value returns
            fbx_data = data['fbx']
            value_col = [col for col in fbx_data.columns if 'Value' in col or 'Close' in col][0]
            target = fbx_data[value_col].pct_change().dropna()
        
        # Align with features
        common_index = self.features.index.intersection(target.index)
        return target.loc[common_index]
    
    def _split_data(self):
        """Split data into train/validation/test sets"""
        # Align features and target
        common_index = self.features.index.intersection(self.target.index)
        self.features = self.features.loc[common_index]
        self.target = self.target.loc[common_index]
        
        # Time series split
        n_samples = len(self.features)
        train_size = int(n_samples * self.ml_config['training']['train_split'])
        val_size = int(n_samples * self.ml_config['training']['validation_split'])
        
        # Split indices
        train_end = train_size
        val_end = train_end + val_size
        
        # Create splits
        self.X_train = self.features.iloc[:train_end]
        self.X_val = self.features.iloc[train_end:val_end]
        self.X_test = self.features.iloc[val_end:]
        
        self.y_train = self.target.iloc[:train_end]
        self.y_val = self.target.iloc[train_end:val_end]
        self.y_test = self.target.iloc[val_end:]
        
        self.logger.info(f"Data split - Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")
    
    def _scale_features(self):
        """Scale features using standard scaler"""
        self.X_train = self.feature_engineer.scale_features(self.X_train, method='standard')
        self.X_val = self.feature_engineer.scale_features(self.X_val, method='standard')
        self.X_test = self.feature_engineer.scale_features(self.X_test, method='standard')
    
    def _optimize_hyperparameters(self):
        """Optimize hyperparameters for all models"""
        enabled_models = [name for name, enabled in self.ml_config['models'].items() if enabled]
        
        optimization_results = self.hyperopt_optimizer.optimize_all_models(
            enabled_models, self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        self.logger.info(f"Hyperparameter optimization completed for {len(optimization_results)} models")
    
    def _train_models(self) -> Dict[str, Any]:
        """Train all enabled models"""
        training_results = self.model_trainer.train_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        self.trained_models = training_results['models']
        self.best_model = training_results['best_model']
        
        return {
            'models_trained': len(self.trained_models),
            'best_model': self.best_model
        }
    
    def _create_ensembles(self) -> Dict[str, Any]:
        """Create ensemble models"""
        ensemble_results = {'ensembles_created': 0, 'best_ensemble': None}
        
        if len(self.trained_models) < 2:
            self.logger.warning("Need at least 2 models for ensemble creation")
            return ensemble_results
        
        # Create different ensemble types
        if self.ml_config['ensemble']['voting']:
            voting_ensemble = self.ensemble_manager.create_voting_ensemble(
                self.trained_models, self.X_val, self.y_val
            )
            if voting_ensemble:
                ensemble_results['ensembles_created'] += 1
        
        if self.ml_config['ensemble']['stacking']:
            stacking_ensemble = self.ensemble_manager.create_stacking_ensemble(
                self.trained_models, self.X_train, self.y_train, self.X_val, self.y_val
            )
            if stacking_ensemble:
                ensemble_results['ensembles_created'] += 1
        
        if self.ml_config['ensemble']['blending']:
            blending_ensemble = self.ensemble_manager.create_blending_ensemble(
                self.trained_models, self.X_val, self.y_val
            )
            if blending_ensemble:
                ensemble_results['ensembles_created'] += 1
        
        # Get best ensemble
        best_ensemble_name, best_ensemble = self.ensemble_manager.get_best_ensemble(
            self.X_test, self.y_test
        )
        
        if best_ensemble:
            ensemble_results['best_ensemble'] = best_ensemble_name
            self.ensemble_models = self.ensemble_manager.ensemble_models
        
        return ensemble_results
    
    def _evaluate_models(self) -> Dict[str, Any]:
        """Evaluate all models on test set"""
        evaluation_results = {}
        
        # Evaluate single models
        for model_name, model in self.trained_models.items():
            try:
                y_pred = model.predict(self.X_test)
                
                evaluation_results[model_name] = {
                    'r2_score': self._calculate_r2(self.y_test, y_pred),
                    'mse': self._calculate_mse(self.y_test, y_pred),
                    'mae': self._calculate_mae(self.y_test, y_pred)
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        # Evaluate ensemble models
        ensemble_results = self.ensemble_manager.evaluate_ensembles(self.X_test, self.y_test)
        evaluation_results.update(ensemble_results)
        
        self.model_performances = evaluation_results
        return evaluation_results
    
    def _generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions for different horizons"""
        prediction_results = {}
        
        # Current predictions (test set)
        prediction_results['test_predictions'] = {}
        
        # Single model predictions
        for model_name, model in self.trained_models.items():
            try:
                y_pred = model.predict(self.X_test)
                prediction_results['test_predictions'][model_name] = {
                    'predictions': y_pred.tolist(),
                    'dates': self.X_test.index.tolist()
                }
            except Exception as e:
                self.logger.error(f"Error generating predictions for {model_name}: {e}")
                continue
        
        # Ensemble predictions
        for ensemble_name, ensemble in self.ensemble_models.items():
            try:
                y_pred = ensemble.predict(self.X_test)
                prediction_results['test_predictions'][f'ensemble_{ensemble_name}'] = {
                    'predictions': y_pred.tolist(),
                    'dates': self.X_test.index.tolist()
                }
            except Exception as e:
                self.logger.error(f"Error generating predictions for ensemble {ensemble_name}: {e}")
                continue
        
        # Future predictions (if requested)
        if self.ml_config['training']['prediction_horizon']:
            prediction_results['future_predictions'] = self._generate_future_predictions()
        
        return prediction_results
    
    def _generate_future_predictions(self) -> Dict[str, Any]:
        """Generate future predictions"""
        future_predictions = {}
        
        # Use last available data for prediction
        last_features = self.X_test.iloc[-1:].copy()
        
        # Generate predictions for different horizons
        for horizon in self.ml_config['training']['prediction_horizon']:
            horizon_predictions = {}
            
            # Single model predictions
            for model_name, model in self.trained_models.items():
                try:
                    pred = model.predict(last_features)[0]
                    horizon_predictions[model_name] = pred
                except Exception as e:
                    self.logger.error(f"Error generating future prediction for {model_name}: {e}")
                    continue
            
            # Ensemble predictions
            for ensemble_name, ensemble in self.ensemble_models.items():
                try:
                    pred = ensemble.predict(last_features)[0]
                    horizon_predictions[f'ensemble_{ensemble_name}'] = pred
                except Exception as e:
                    self.logger.error(f"Error generating future prediction for ensemble {ensemble_name}: {e}")
                    continue
            
            future_predictions[f'{horizon}_day_ahead'] = horizon_predictions
        
        return future_predictions
    
    def _calculate_r2(self, y_true, y_pred):
        """Calculate R² score"""
        try:
            from sklearn.metrics import r2_score
            return r2_score(y_true, y_pred)
        except:
            return 0.0
    
    def _calculate_mse(self, y_true, y_pred):
        """Calculate MSE"""
        try:
            from sklearn.metrics import mean_squared_error
            return mean_squared_error(y_true, y_pred)
        except:
            return np.inf
    
    def _calculate_mae(self, y_true, y_pred):
        """Calculate MAE"""
        try:
            from sklearn.metrics import mean_absolute_error
            return mean_absolute_error(y_true, y_pred)
        except:
            return np.inf
    
    def _save_ml_results(self):
        """Save ML results to disk"""
        import joblib
        
        # Save models
        models_dir = os.path.join(self.ml_output_dir, "models")
        self.model_trainer.save_models(models_dir)
        
        # Save ensembles
        ensembles_dir = os.path.join(self.ml_output_dir, "ensembles")
        self.ensemble_manager.save_ensembles(ensembles_dir)
        
        # Save optimization results
        hyperopt_dir = os.path.join(self.ml_output_dir, "hyperopt")
        self.hyperopt_optimizer.save_optimization_results(hyperopt_dir)
        
        # Save feature engineering results
        features_dir = os.path.join(self.ml_output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        
        if self.features is not None:
            self.features.to_csv(os.path.join(features_dir, "engineered_features.csv"))
        
        if self.feature_engineer.feature_names:
            joblib.dump(self.feature_engineer.feature_names, os.path.join(features_dir, "feature_names.pkl"))
        
        if self.feature_engineer.selected_features:
            joblib.dump(self.feature_engineer.selected_features, os.path.join(features_dir, "selected_features.pkl"))
        
        # Save overall results
        overall_results = {
            'model_performances': self.model_performances,
            'predictions': self.predictions,
            'best_model': self.best_model,
            'feature_names': self.feature_engineer.feature_names,
            'selected_features': self.feature_engineer.selected_features
        }
        
        joblib.dump(overall_results, os.path.join(self.ml_output_dir, "ml_results.pkl"))
        
        self.logger.info(f"ML results saved to {self.ml_output_dir}")
    
    def get_ml_summary(self) -> Dict[str, Any]:
        """Get comprehensive ML summary"""
        summary = {
            'ml_enabled': self.config.ML_ENABLED,
            'models_trained': len(self.trained_models),
            'ensembles_created': len(self.ensemble_models),
            'features_engineered': len(self.features.columns) if self.features is not None else 0,
            'features_selected': len(self.feature_engineer.selected_features),
            'best_single_model': self.best_model,
            'model_performances': self.model_performances,
            'data_split': {
                'train_size': len(self.X_train) if self.X_train is not None else 0,
                'val_size': len(self.X_val) if self.X_val is not None else 0,
                'test_size': len(self.X_test) if self.X_test is not None else 0
            }
        }
        
        # Add best ensemble
        if self.ensemble_models:
            best_ensemble_name, _ = self.ensemble_manager.get_best_ensemble(self.X_test, self.y_test)
            summary['best_ensemble'] = best_ensemble_name
        
        return summary
    
    def predict(self, data: Dict[str, pd.DataFrame], model_name: str = None) -> np.ndarray:
        """
        Make predictions using trained models
        
        Args:
            data: Input data dictionary
            model_name: Specific model to use (if None, uses best model)
            
        Returns:
            Array of predictions
        """
        # Engineer features for new data
        features = self.feature_engineer.engineer_features(data)
        
        # Select same features as training
        if self.feature_engineer.selected_features:
            features = features[self.feature_engineer.selected_features]
        
        # Scale features
        features = self.feature_engineer.scale_features(features, method='standard')
        
        # Make predictions
        if model_name is None:
            model_name = self.best_model
        
        if model_name in self.trained_models:
            return self.trained_models[model_name].predict(features)
        elif f'ensemble_{model_name}' in self.ensemble_models:
            return self.ensemble_models[f'ensemble_{model_name}'].predict(features)
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def get_feature_importance(self, model_name: str = None) -> Optional[pd.Series]:
        """Get feature importance for a specific model"""
        if model_name is None:
            model_name = self.best_model
        
        return self.model_trainer.get_feature_importance(model_name)
