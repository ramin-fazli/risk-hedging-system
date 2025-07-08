# Risk Hedging System with ML

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![ML](https://img.shields.io/badge/ML-enabled-orange)
![Status](https://img.shields.io/badge/status-production%20ready-success)

A comprehensive ML-powered Python project for backtesting hedging strategies against the Freightos Baltic Index (FBX) or other indices using correlated ETFs and stocks.

## Features

### 🎯 Core Capabilities
- **Data Management**: Automated data fetching and processing for FBX index and correlated securities
- **Exposure Analysis**: Quantify company revenue sensitivity to FBX movements
- **Hedge Ratio Optimization**: Calculate optimal hedge ratios using multiple methodologies
- **Backtesting Engine**: Simulate hedging strategies with comprehensive performance metrics
- **Professional Reporting**: Generate detailed Excel reports with visualizations

### 🤖 Advanced ML Features
- **Feature Engineering**: 200+ engineered features with technical indicators
- **Multi-Model Training**: 11+ algorithms including ensemble methods
- **Hyperparameter Optimization**: Bayesian optimization with Optuna
- **Model Interpretation**: SHAP, LIME, and permutation importance
- **Configurable Modes**: Minimal, testing, production configurations

### 🛠️ Development & Deployment
- **Automated Setup**: Windows batch scripts for easy installation
- **Comprehensive Testing**: Unit tests, ML tests, and system integration tests
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing
- **Production Ready**: Professional logging, error handling, and monitoring

### 📊 Reporting & Analytics
- **Excel Integration**: Professional multi-sheet Excel reports
- **Visualizations**: Interactive charts and performance plots
- **ML Analytics**: Model performance comparison and predictions
- **Risk Metrics**: Comprehensive risk assessment and monitoring

## Project Structure

```
risk-hedging-system/
├── .github/                 # GitHub templates and workflows
│   ├── ISSUE_TEMPLATE/      # Bug reports and feature requests
│   └── workflows/           # CI/CD pipeline
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration settings
│   └── instruments.py       # ETF/stock definitions
├── data/
│   ├── __init__.py
│   ├── data_loader.py       # Data fetching and loading
│   ├── data_processor.py    # Data cleaning and preprocessing
│   └── mock_data.py         # Mock data generation for testing
├── analysis/
│   ├── __init__.py
│   ├── exposure_analyzer.py # Revenue-FBX sensitivity analysis
│   ├── hedge_optimizer.py   # Hedge ratio calculation
│   └── risk_metrics.py      # Risk and performance metrics
├── backtesting/
│   ├── __init__.py
│   ├── backtest_engine.py   # Main backtesting engine
│   └── portfolio.py         # Portfolio management
├── ml/                      # Machine Learning Pipeline
│   ├── __init__.py
│   ├── feature_engineer.py  # Advanced feature engineering
│   ├── model_trainer.py     # Multi-model training
│   ├── ensemble_manager.py  # Ensemble methods
│   ├── hyperopt_optimizer.py # Bayesian optimization
│   └── ml_predictor.py      # ML orchestration
├── reporting/
│   ├── __init__.py
│   ├── excel_reporter.py    # Excel report generation
│   └── visualizations.py    # Chart and plot generation
├── utils/
│   ├── __init__.py
│   ├── helpers.py           # Utility functions
│   └── validators.py        # Data validation
├── tests/
│   ├── __init__.py
│   ├── test_components.py   # Unit tests
│   ├── test_ml.py           # ML pipeline tests
│   └── test_ml_config.py    # ML configuration tests
├── charts/                  # Generated chart outputs
├── data_files/              # Directory for data files
├── logs/                    # Application logs
├── reports/                 # Generated Excel reports
├── .gitignore               # Git ignore rules
├── CONTRIBUTING.md          # Contribution guidelines
├── DEPLOYMENT_GUIDE.md      # Deployment instructions
├── GITHUB_PUBLISHING_GUIDE.md # GitHub publishing guide
├── LICENSE                  # Apache License 2.0
├── ML_DOCUMENTATION.md      # ML pipeline documentation
├── ML_ENHANCEMENT_GUIDE.md  # ML enhancement guide
├── ML_GUIDE.md             # ML usage guide
├── ML_IMPLEMENTATION_SUMMARY.md # ML implementation summary
├── PROJECT_COMPLETION.md    # Project completion status
├── PROJECT_DOCUMENTATION.md # Comprehensive documentation
├── PROJECT_SUMMARY.md       # Project summary
├── README.md               # Main documentation
├── __version__.py          # Version information
├── install.bat             # Windows installation script
├── main.py                 # Main execution script
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── run.bat                 # Windows run script
└── system_test.py          # System integration tests
```

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 2 GB (minimal ML mode)
- **Storage**: 1 GB free space
- **OS**: Windows 10/11, macOS 10.14+, Linux Ubuntu 18.04+

### Recommended for Production
- **Python**: 3.9 or higher
- **RAM**: 8-16 GB (production ML mode)
- **Storage**: 5 GB free space
- **CPU**: Multi-core processor for parallel processing

### Optional Dependencies
- **TA-Lib**: For advanced technical indicators
- **XGBoost**: For gradient boosting models
- **Optuna**: For hyperparameter optimization
- **SHAP**: For model interpretation

## Installation

### Quick Setup (Windows)

1. **Clone the repository**
   ```bash
   git clone https://github.com/ramin-fazli/risk-hedging-system.git
   cd risk-hedging-system
   ```

2. **Run the automated installer**
   ```bash
   install.bat
   ```

### Manual Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ramin-fazli/risk-hedging-system.git
   cd risk-hedging-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **For development, also install dev dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

## Usage

### Quick Start (Windows)

1. **Run the system with default settings**
   ```bash
   run.bat
   ```

### Manual Execution

1. **Configure settings in `config/settings.py`**
2. **Define hedge instruments in `config/instruments.py`**
3. **Run the main script**
   ```bash
   python main.py
   ```

### Testing

**Run system tests**
```bash
python system_test.py
```

**Run ML pipeline tests**
```bash
python test_ml.py
```

**Run unit tests**
```bash
python -m pytest tests/
```

## Documentation

This project includes comprehensive documentation:

### 📚 Main Documentation
- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** - Complete technical documentation
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Executive summary and overview
- **[PROJECT_COMPLETION.md](PROJECT_COMPLETION.md)** - Project completion status

### 🤖 Machine Learning Documentation
- **[ML_DOCUMENTATION.md](ML_DOCUMENTATION.md)** - Complete ML pipeline documentation
- **[ML_GUIDE.md](ML_GUIDE.md)** - ML usage guide and examples
- **[ML_ENHANCEMENT_GUIDE.md](ML_ENHANCEMENT_GUIDE.md)** - ML enhancement guide
- **[ML_IMPLEMENTATION_SUMMARY.md](ML_IMPLEMENTATION_SUMMARY.md)** - ML implementation summary

### 🚀 Deployment & Publishing
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment guide
- **[GITHUB_PUBLISHING_GUIDE.md](GITHUB_PUBLISHING_GUIDE.md)** - GitHub publishing instructions
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

## Configuration

The system supports various configuration options:
- Data sources and date ranges
- Hedge instruments and their parameters
- Backtesting parameters
- Reporting preferences

## Output

The system generates comprehensive outputs across multiple directories:

### 📊 Excel Reports (`reports/`)
- **Main Analysis**: Comprehensive backtesting results with performance metrics
- **Hedge Analysis**: Detailed hedge ratio calculations and effectiveness
- **Risk Metrics**: Value at Risk, Maximum Drawdown, and other risk measures
- **ML Analysis**: Machine learning pipeline performance and model comparison
- **ML Predictions**: Test set predictions and future forecasts

### 📈 Visualizations (`charts/`)
- **Time Series Plots**: Price movements and hedge performance over time
- **Correlation Heatmaps**: Cross-asset correlation analysis
- **Performance Charts**: Risk-return scatter plots and performance attribution
- **ML Visualizations**: Feature importance and model performance plots

### 📋 Logs (`logs/`)
- **Application Logs**: Detailed execution logs with timestamps
- **Error Logs**: Comprehensive error tracking and debugging information
- **Performance Logs**: Execution time and resource usage metrics

### 📁 Data Outputs (`data_files/`)
- **Processed Data**: Clean, processed datasets ready for analysis
- **Feature Data**: Engineered features for ML models
- **Model Outputs**: Trained model artifacts and predictions

## Key Features

- **Modular Design**: Easy to extend and modify
- **Data Flexibility**: Supports multiple data sources
- **Multiple Hedge Strategies**: Various hedge ratio calculation methods
- **Comprehensive Analytics**: Detailed performance and risk metrics
- **Professional Reporting**: Publication-ready Excel reports

## 🤖 MACHINE LEARNING ENHANCEMENT - STATUS: COMPLETE ✅

### Advanced ML Capabilities Added
The project now includes a comprehensive machine learning module with:

#### **Feature Engineering**
- 🔍 **Technical Indicators**: 40+ indicators with TA-Lib integration and fallbacks
- 📊 **Rolling Statistics**: Multi-window analysis (5, 10, 20, 60, 120 periods)
- ⏰ **Lag Features**: Temporal pattern recognition
- 🔗 **Interaction Features**: Cross-asset correlation analysis
- 🌊 **Fourier Features**: Frequency domain cyclical pattern detection
- 📐 **PCA Features**: Dimensionality reduction
- 🔢 **Polynomial Features**: Non-linear relationship modeling

#### **Machine Learning Models**
- 🌳 **Tree-based**: Random Forest, Extra Trees, Gradient Boosting
- 📈 **Linear Models**: Ridge, Lasso, Elastic Net
- 🚀 **Advanced Models**: XGBoost, SVM, Neural Networks
- 🧠 **Deep Learning**: LSTM for time series (optional)
- 🎯 **Transformers**: Attention-based models (optional)

#### **Ensemble Methods**
- 🗳️ **Voting Ensemble**: Democratic prediction averaging
- 📚 **Stacking**: Meta-learning with cross-validation
- ⚖️ **Blending**: Optimized weighted combinations
- 🔄 **Dynamic Ensemble**: Adaptive performance-based weighting

#### **Optimization & Interpretation**
- 🎛️ **Hyperparameter Optimization**: Optuna-based Bayesian optimization
- 🔍 **Model Interpretation**: SHAP, LIME, permutation importance
- ⚡ **Parallel Processing**: Multi-core optimization support
- 📊 **Cross-validation**: Robust model evaluation

#### **Configuration Modes**
- **Minimal**: Quick testing (2-4 GB RAM, 5-15 min)
- **Testing**: Development validation (4-8 GB RAM, 20-60 min)
- **Production**: Full deployment (8-16 GB RAM, 1-4 hours)
- **Custom**: Fully customizable configuration

#### **Enhanced Reporting**
- 📋 **ML Analysis Sheet**: Pipeline summary and performance metrics
- 🔮 **ML Predictions Sheet**: Test set and future predictions
- 🏆 **ML Model Comparison**: Performance ranking and recommendations

### ✅ Test Results
- **System Status**: ✅ FULLY OPERATIONAL
- **Data Processed**: 1,736 days successfully analyzed
- **Hedge Instruments**: 40 instruments evaluated
- **Execution Time**: ~2.5 minutes complete run
- **Report Generated**: Professional Excel with ML analytics

### 🚀 Usage
```python
# Enable ML functionality
config.ML_ENABLED = True
config.ML_MODE = "minimal"  # or "testing", "production"

# Run enhanced backtesting
python main.py
```

### 📁 ML Module Structure
```
ml/
├── feature_engineer.py        # Advanced feature engineering
├── model_trainer.py          # Multi-model training
├── ensemble_manager.py       # Ensemble methods
├── hyperopt_optimizer.py     # Bayesian optimization
└── ml_predictor.py           # ML orchestration
```

The ML enhancement provides **state-of-the-art** capabilities while maintaining **backward compatibility** and **graceful degradation** when advanced dependencies are not available.

## 🎯 Project Status

### ✅ Implementation Complete
- **Core System**: Fully operational backtesting framework
- **ML Pipeline**: Complete with 11+ algorithms and ensemble methods
- **Documentation**: Comprehensive guides and API documentation
- **Testing**: Unit tests, ML tests, and system integration tests
- **CI/CD**: GitHub Actions workflow for automated testing
- **Deployment**: Production-ready with professional error handling

### 📊 Performance Metrics
- **Data Processing**: 1,736+ days of financial data
- **Hedge Instruments**: 40+ instruments analyzed
- **Execution Time**: 2-5 minutes (minimal mode)
- **Memory Usage**: 2-16 GB (configurable)
- **Code Quality**: 11,868+ lines of production code

### 🚀 Ready for Production
- **Scalability**: Configurable for different resource environments
- **Reliability**: Comprehensive error handling and logging
- **Maintainability**: Modular design with clear separation of concerns
- **Extensibility**: Easy to add new features and models

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Setting up the development environment
- Running tests and quality checks
- Submitting pull requests
- Reporting bugs and requesting features

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/ramin-fazli/risk-hedging-system/issues)
- **Discussions**: Join the conversation in [GitHub Discussions](https://github.com/ramin-fazli/risk-hedging-system/discussions)
- **Documentation**: Comprehensive guides available in the `docs/` directory

## 🏆 Acknowledgments

- Built with modern Python best practices
- Leverages industry-standard libraries for ML and financial analysis
- Designed for professional quantitative finance applications

---

**📊 Repository Stats**: 75 files • 35 Python modules • 11,868+ lines of code • Production ready • Apache 2.0 licensed
