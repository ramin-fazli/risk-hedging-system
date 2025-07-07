# Machine Learning Module Documentation

## Overview

The FBX Hedging Strategy system now includes advanced machine learning capabilities that can significantly enhance prediction accuracy and hedge optimization. The ML module provides a comprehensive framework for feature engineering, model training, ensemble methods, and hyperparameter optimization.

## Features

### 1. Advanced Feature Engineering
- **Technical Indicators**: 40+ technical indicators using TA-Lib
- **Rolling Statistics**: Multi-window statistical features
- **Lag Features**: Time-lagged variables for temporal patterns
- **Interaction Features**: Cross-asset correlations and ratios
- **Fourier Features**: Frequency domain analysis for cyclical patterns
- **Wavelet Features**: Time-frequency analysis (optional)
- **PCA Features**: Dimensionality reduction
- **Polynomial Features**: Non-linear feature combinations

### 2. Multiple ML Models
- **Tree-based Models**: Random Forest, Extra Trees, Gradient Boosting
- **Linear Models**: Ridge, Lasso, Elastic Net
- **Advanced Models**: XGBoost, SVM, Neural Networks
- **Deep Learning**: LSTM for time series (optional)
- **Transformers**: Attention-based models (optional)

### 3. Ensemble Methods
- **Voting Ensemble**: Simple averaging of predictions
- **Stacking**: Meta-learning approach
- **Blending**: Weighted combination optimization
- **Dynamic Ensemble**: Adaptive weighting based on performance

### 4. Hyperparameter Optimization
- **Optuna**: Bayesian optimization with pruning
- **Parallel Processing**: Multi-core optimization
- **Early Stopping**: Efficient resource usage
- **Cross-validation**: Robust parameter selection

### 5. Model Interpretation
- **SHAP**: Shapley values for feature importance
- **LIME**: Local interpretable model-agnostic explanations
- **Permutation Importance**: Feature importance ranking
- **Partial Dependence**: Feature effect visualization

## Configuration Modes

### Minimal Mode (`ML_MODE = "minimal"`)
**Best for**: Quick testing, development, resource-constrained environments

**Features**:
- 2 models: Random Forest, Ridge Regression
- Basic feature engineering
- Simple voting ensemble
- 20 hyperparameter trials
- 10-minute timeout

**Resource Requirements**:
- RAM: 2-4 GB
- CPU: 2-4 cores
- Time: 5-15 minutes

### Testing Mode (`ML_MODE = "testing"`)
**Best for**: Development, model validation, moderate testing

**Features**:
- 6 models: Random Forest, XGBoost, Ridge, Lasso, Elastic Net, Gradient Boosting
- Advanced feature engineering
- Multiple ensemble methods
- 100 hyperparameter trials
- 30-minute timeout

**Resource Requirements**:
- RAM: 4-8 GB
- CPU: 4-8 cores
- Time: 20-60 minutes

### Production Mode (`ML_MODE = "production"`)
**Best for**: Full deployment, maximum accuracy, comprehensive analysis

**Features**:
- 11 models including LSTM and Neural Networks
- Complete feature engineering pipeline
- All ensemble methods
- 500 hyperparameter trials
- 2-hour timeout

**Resource Requirements**:
- RAM: 8-16 GB
- CPU: 8+ cores
- Time: 1-4 hours

### Custom Mode (`ML_MODE = "custom"`)
**Best for**: Specific requirements, fine-tuned configurations

**Features**:
- Fully customizable configuration
- All options available
- Manual parameter tuning

## Configuration

### Basic Setup
```python
# config/settings.py
config = Config()
config.ML_ENABLED = True
config.ML_MODE = "minimal"  # or "testing", "production", "custom"
```

### Advanced Configuration
```python
# For custom mode
config.ML_CONFIG = {
    "models": {
        "random_forest": True,
        "xgboost": True,
        "lstm": False,  # Disable for faster execution
        # ... other models
    },
    "hyperopt": {
        "n_trials": 50,
        "timeout": 1800,  # 30 minutes
    },
    # ... other settings
}
```

## Usage

### 1. Enable ML in Configuration
```python
from config.settings import Config

config = Config()
config.ML_ENABLED = True
config.ML_MODE = "minimal"  # Choose appropriate mode
```

### 2. Run with ML
```python
# The ML pipeline runs automatically when enabled
python main.py
```

### 3. Access ML Results
The ML results are included in the Excel report with dedicated worksheets:
- **ML Analysis**: Pipeline summary and model performance
- **ML Predictions**: Test set and future predictions
- **ML Model Comparison**: Performance ranking and recommendations

## Performance Metrics

### Model Evaluation
- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

### Ensemble Performance
- **Voting**: Simple average performance
- **Stacking**: Meta-model performance
- **Blending**: Optimized weight performance

## Best Practices

### 1. Start Simple
- Begin with "minimal" mode for testing
- Gradually increase complexity
- Monitor resource usage

### 2. Data Quality
- Ensure sufficient historical data (minimum 2-3 years)
- Check for data gaps or anomalies
- Validate data consistency

### 3. Feature Engineering
- Review feature importance regularly
- Remove highly correlated features
- Consider domain expertise

### 4. Model Selection
- Use cross-validation for reliable estimates
- Consider ensemble methods for robustness
- Balance complexity vs. interpretability

### 5. Hyperparameter Optimization
- Set reasonable timeouts
- Use parallel processing when available
- Monitor convergence

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce ML_MODE to "minimal"
   - Decrease max_features in feature_selection
   - Disable heavy models (LSTM, Transformers)

2. **Long Execution Times**
   - Reduce n_trials in hyperopt
   - Decrease timeout values
   - Use fewer models

3. **Poor Model Performance**
   - Increase feature engineering
   - Try different models
   - Check data quality

4. **Import Errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python version compatibility
   - Verify package versions

### Dependencies

**Required**:
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- joblib

**Optional (for advanced features)**:
- xgboost
- tensorflow (for LSTM)
- optuna (for hyperparameter optimization)
- ta-lib (for technical indicators)
- shap, lime (for model interpretation)

### Performance Optimization

1. **Parallel Processing**
   - Enable n_jobs=-1 in model configurations
   - Use parallel hyperparameter optimization
   - Consider multi-core systems

2. **Memory Management**
   - Monitor memory usage during execution
   - Use data chunking for large datasets
   - Clear unused variables

3. **Caching**
   - Cache feature engineering results
   - Save trained models for reuse
   - Store hyperparameter optimization results

## Output

### Excel Report Sections
1. **ML Analysis**: Overview of ML pipeline execution
2. **ML Predictions**: Model predictions and forecasts
3. **ML Model Comparison**: Performance comparison and rankings

### Saved Files
- `ml_output/models/`: Trained model files
- `ml_output/ensembles/`: Ensemble model files
- `ml_output/hyperopt/`: Hyperparameter optimization results
- `ml_output/features/`: Feature engineering outputs

## Future Enhancements

### Planned Features
1. **Real-time Prediction API**
2. **Model Monitoring and Drift Detection**
3. **Automated Model Retraining**
4. **Advanced Ensemble Methods**
5. **Deep Learning Enhancements**

### Customization Options
1. **Custom Feature Engineering**
2. **Domain-specific Models**
3. **Risk-aware Ensembles**
4. **Multi-objective Optimization**

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files in `logs/`
3. Examine ML output in `ml_output/`
4. Verify configuration settings

---

**Note**: The ML module is designed to be modular and can be disabled by setting `ML_ENABLED = False` in the configuration. This allows users to run the traditional hedging analysis without ML dependencies if needed.
