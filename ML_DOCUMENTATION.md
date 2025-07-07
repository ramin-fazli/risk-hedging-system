# Machine Learning Enhancement Documentation

## Overview

The FBX Hedging Strategy Backtesting System has been enhanced with advanced machine learning capabilities to provide more accurate predictions and insights. This enhancement includes multiple ML algorithms, ensemble methods, hyperparameter optimization, and comprehensive feature engineering.

## ML Features

### 1. Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Williams %R, CCI
- **Rolling Statistics**: Mean, std, min, max, median, skewness, kurtosis, quantiles
- **Lag Features**: Multiple lag periods (1, 2, 3, 5, 10, 20, 30, 60 days)
- **Interaction Features**: Cross-correlations, betas, ratios between instruments
- **Fourier Features**: Cyclical pattern detection using FFT
- **Wavelet Features**: Time-frequency analysis (optional)
- **PCA Features**: Dimensionality reduction
- **Polynomial Features**: Non-linear relationships

### 2. Machine Learning Models
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting framework
- **LSTM**: Long Short-Term Memory neural networks
- **SVM**: Support Vector Machines
- **Elastic Net**: Regularized linear regression
- **Ridge Regression**: L2 regularized linear regression
- **Lasso Regression**: L1 regularized linear regression
- **Gradient Boosting**: Gradient boosting machines
- **Extra Trees**: Extremely randomized trees
- **Neural Networks**: Multi-layer perceptrons

### 3. Ensemble Methods
- **Voting**: Simple average of model predictions
- **Stacking**: Meta-learner trained on base model predictions
- **Blending**: Weighted combination with optimized weights
- **Dynamic Ensemble**: Adaptive weights based on recent performance

### 4. Hyperparameter Optimization
- **Optuna**: Bayesian optimization framework
- **Automated**: Optimizes all model hyperparameters
- **Configurable**: Number of trials and timeout settings
- **Efficient**: Uses pruning to stop unpromising trials

### 5. Model Validation
- **Time Series Cross-Validation**: Respects temporal order
- **Walk-Forward Validation**: Realistic backtesting approach
- **Multiple Metrics**: R², MSE, MAE, RMSE
- **Performance Tracking**: Comprehensive evaluation

## Configuration

### Enabling/Disabling ML
```python
# In config/settings.py
ML_ENABLED = True  # Set to False to disable ML functionality
```

### Model Configuration
```python
ML_CONFIG = {
    'models': {
        'random_forest': True,      # Enable/disable specific models
        'xgboost': True,
        'lstm': True,
        'svm': True,
        'elastic_net': True,
        'ridge': True,
        'lasso': True,
        'gradient_boosting': True,
        'extra_trees': True,
        'neural_network': True
    }
}
```

### Feature Engineering Configuration
```python
'feature_engineering': {
    'technical_indicators': True,
    'rolling_statistics': True,
    'lag_features': True,
    'interaction_features': True,
    'fourier_features': True,
    'wavelet_features': False,  # Computationally intensive
    'pca_features': True,
    'polynomial_features': True
}
```

### Ensemble Configuration
```python
'ensemble': {
    'voting': True,
    'stacking': True,
    'blending': True,
    'bayesian_optimization': True
}
```

### Hyperparameter Optimization
```python
'hyperopt': {
    'method': 'optuna',
    'n_trials': 100,
    'parallel': True,
    'pruning': True,
    'timeout': 3600  # 1 hour timeout
}
```

## Usage

### Basic Usage
The ML functionality is automatically integrated into the main backtesting system. Simply run:

```bash
python main.py
```

### Advanced Usage
For more control, you can use the ML components directly:

```python
from ml.ml_predictor import MLPredictor
from config.settings import Config

config = Config()
ml_predictor = MLPredictor(config)

# Run full ML pipeline
ml_results = ml_predictor.run_full_ml_pipeline(processed_data)

# Make predictions
predictions = ml_predictor.predict(new_data)
```

## Output and Results

### ML Results Structure
```python
ml_results = {
    'status': 'completed',
    'models_trained': 8,
    'ensembles_created': 3,
    'best_single_model': 'random_forest',
    'best_ensemble': 'stacking',
    'model_performances': {
        'random_forest': {'r2_score': 0.65, 'mse': 0.001},
        'xgboost': {'r2_score': 0.68, 'mse': 0.0009},
        # ... other models
    },
    'predictions': {
        'test_predictions': {...},
        'future_predictions': {...}
    }
}
```

### Excel Report Integration
The ML results are automatically integrated into the Excel report with new worksheets:
- **ML Analysis**: Summary of ML pipeline and model performances
- **ML Predictions**: Test set and future predictions
- **ML Model Comparison**: Detailed comparison of all models

### Files Generated
- `ml_output/models/`: Trained model files
- `ml_output/ensembles/`: Ensemble model files
- `ml_output/hyperopt/`: Hyperparameter optimization results
- `ml_output/features/`: Feature engineering results
- `ml_output/ml_results.pkl`: Overall ML results

## Performance Considerations

### Computational Requirements
- **Memory**: 4-8 GB RAM recommended
- **CPU**: Multi-core processor for parallel processing
- **Storage**: ~1 GB for model files and results

### Optimization Settings
For faster execution, consider:
- Reducing number of hyperparameter trials
- Disabling computationally intensive features (wavelet, transformer models)
- Using fewer models for initial testing

```python
# Quick test configuration
config.ML_CONFIG['hyperopt']['n_trials'] = 10
config.ML_CONFIG['models']['lstm'] = False
config.ML_CONFIG['models']['neural_network'] = False
config.ML_CONFIG['feature_engineering']['wavelet_features'] = False
```

## Best Practices

### 1. Data Quality
- Ensure clean, consistent data
- Handle missing values appropriately
- Consider data stationarity for time series

### 2. Feature Selection
- Start with basic features and gradually add complexity
- Monitor feature importance to identify key predictors
- Consider domain knowledge in feature engineering

### 3. Model Selection
- Start with simpler models (linear regression, random forest)
- Gradually add complexity based on performance improvements
- Use ensemble methods for better generalization

### 4. Validation
- Always use time series cross-validation
- Monitor for overfitting
- Validate on out-of-sample data

### 5. Interpretation
- Review feature importance
- Understand model predictions
- Consider model explainability (SHAP, LIME)

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Solution: Reduce dataset size or disable memory-intensive features

2. **Slow Performance**
   - Solution: Reduce hyperparameter trials or disable complex models

3. **Import Errors**
   - Solution: Install missing dependencies with `pip install -r requirements.txt`

4. **Poor Model Performance**
   - Solution: Review feature engineering, data quality, or model selection

### Dependencies
Make sure all required packages are installed:
```bash
pip install -r requirements.txt
```

Key ML dependencies:
- scikit-learn>=1.3.0
- xgboost>=1.7.0
- tensorflow>=2.13.0
- optuna>=3.0.0
- ta-lib>=0.4.0

## Future Enhancements

Potential improvements for future versions:
1. **Transformer Models**: Attention-based models for sequence prediction
2. **Reinforcement Learning**: Adaptive hedging strategies
3. **Online Learning**: Real-time model updates
4. **Advanced Feature Engineering**: Alternative data sources
5. **Model Interpretability**: Enhanced explainability tools

## Contact and Support

For questions or issues related to the ML functionality, please refer to the main project documentation or contact the development team.
