# FBX Hedging Strategy Backtesting System
# Complete Project Documentation

## Overview
This comprehensive Python system backtests hedging strategies against the Freightos Baltic Index (FBX) to help companies hedge their revenue exposure to shipping costs fluctuations.

## Key Features

### 1. Data Management
- **Automated Data Loading**: Fetches FBX index data, hedge instruments (ETFs/stocks), and revenue data
- **Mock Data Generation**: Creates realistic synthetic data for testing when real data is unavailable
- **Data Validation**: Comprehensive data quality checks and validation
- **Multiple Data Sources**: Supports Yahoo Finance, CSV files, and API integration

### 2. Exposure Analysis
- **Revenue Sensitivity Analysis**: Quantifies how company revenue responds to FBX movements
- **Correlation Analysis**: Measures relationships between FBX and potential hedge instruments
- **Statistical Testing**: P-values, confidence intervals, and significance testing
- **Time-Varying Analysis**: Rolling correlations and regime-based analysis

### 3. Hedge Optimization
- **Multiple Methods**: OLS regression, minimum variance, correlation-based, and dynamic beta
- **Optimal Ratio Selection**: Automatically selects best hedge ratios based on effectiveness
- **Risk Management**: Position sizing with maximum exposure limits
- **Ensemble Approach**: Combines multiple methods for robust results

### 4. Backtesting Engine
- **Comprehensive Simulation**: Tests hedged vs unhedged strategies over historical periods
- **Portfolio Management**: Realistic trading simulation with transaction costs
- **Rebalancing**: Configurable rebalancing frequencies (daily, weekly, monthly)
- **Performance Tracking**: Detailed P&L attribution and performance metrics

### 5. Risk Analysis
- **Value at Risk (VaR)**: Parametric and historical VaR calculations
- **Stress Testing**: Performance under extreme market conditions
- **Scenario Analysis**: Bull/bear markets, high/low volatility regimes
- **Tail Risk Metrics**: Expected shortfall, extreme value analysis

### 6. Professional Reporting
- **Excel Reports**: Comprehensive multi-worksheet reports with formatting
- **Executive Summary**: Key findings and recommendations
- **Detailed Analytics**: In-depth analysis of all components
- **Visualizations**: Charts and graphs for performance comparison

## Project Structure

```
shipping_project/
├── config/                 # Configuration files
│   ├── settings.py         # Main configuration
│   └── instruments.py      # Hedge instruments definition
├── data/                   # Data handling
│   ├── data_loader.py      # Data fetching and loading
│   ├── data_processor.py   # Data cleaning and processing
│   └── mock_data.py        # Mock data generation
├── analysis/               # Core analysis modules
│   ├── exposure_analyzer.py # Revenue-FBX relationship analysis
│   ├── hedge_optimizer.py  # Hedge ratio optimization
│   └── risk_metrics.py     # Risk and performance metrics
├── backtesting/            # Backtesting engine
│   ├── backtest_engine.py  # Main backtesting logic
│   └── portfolio.py        # Portfolio management
├── reporting/              # Report generation
│   ├── excel_reporter.py   # Excel report generation
│   └── visualizations.py   # Chart and plot generation
├── utils/                  # Utility functions
│   ├── helpers.py          # Helper functions
│   └── validators.py       # Data validation
├── tests/                  # Unit tests
│   └── test_components.py  # Test suite
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
├── install.bat            # Windows installation script
└── run.bat                # Windows execution script
```

## Installation and Usage

### Prerequisites
- Python 3.8 or higher
- Windows PowerShell or Command Prompt

### Installation Steps

1. **Clone or Download**: Get the project files
2. **Run Installation**: Execute `install.bat` (Windows) or manually:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Create Directories**: The system will create necessary directories automatically

### Configuration

Edit `config/settings.py` to customize:
- **Date Range**: `START_DATE` and `END_DATE`
- **Capital**: `INITIAL_CAPITAL` for backtesting
- **Risk Parameters**: `MAX_POSITION_SIZE`, `TRANSACTION_COST`
- **Rebalancing**: `REBALANCE_FREQUENCY`

Edit `config/instruments.py` to modify hedge instruments:
- **ETFs**: SHIP, IYT, FTXD, BDRY
- **Stocks**: DAC, ZIM, MATX, STNG, UPS, FDX
- **Expected Correlations**: Based on historical relationships

### Running the System

#### Method 1: Batch File (Windows)
```bash
run.bat
```

#### Method 2: Manual Execution
```bash
venv\Scripts\activate
python main.py
```

### Output

The system generates:
1. **Excel Report**: Comprehensive analysis in `reports/` directory
2. **Log Files**: Detailed execution logs in `logs/` directory
3. **Charts**: Visual outputs in `charts/` directory (if enabled)

## Key Components Detail

### 1. Exposure Analysis
- **Linear Regression**: Revenue = α + β × FBX + ε
- **Correlation Analysis**: Pearson correlation with confidence intervals
- **Beta Calculation**: Revenue sensitivity to FBX changes
- **Statistical Tests**: P-values, F-statistics, R-squared

### 2. Hedge Instruments
**ETFs**:
- SHIP: Transportation sector (negative correlation with FBX)
- IYT: Transportation average (negative correlation)
- BDRY: Dry bulk shipping (positive correlation)

**Stocks**:
- Shipping Companies: DAC, ZIM, MATX (positive correlation)
- Logistics Companies: UPS, FDX (negative correlation)

### 3. Hedge Strategies
- **OLS Regression**: Simple linear relationship
- **Minimum Variance**: Optimal variance reduction
- **Correlation-Based**: Uses correlation and volatility ratios
- **Dynamic Beta**: Time-varying hedge ratios

### 4. Performance Metrics
- **Return Metrics**: Total, annualized, Sharpe ratio
- **Risk Metrics**: Volatility, VaR, maximum drawdown
- **Hedge Effectiveness**: Variance reduction, correlation
- **Attribution**: P&L breakdown by component

### 5. Excel Report Structure
1. **Executive Summary**: Key findings and recommendations
2. **Data Summary**: Data quality and coverage
3. **Exposure Analysis**: FBX-revenue relationship
4. **Hedge Ratios**: Optimal ratios and methods
5. **Backtesting Results**: Performance comparison
6. **Risk Analysis**: VaR, stress testing
7. **Scenario Analysis**: Market regime performance
8. **Detailed Data**: Raw data outputs

## Technical Features

### Data Processing
- **Missing Value Handling**: Forward fill, interpolation
- **Outlier Treatment**: IQR-based capping
- **Frequency Alignment**: Resampling to common frequency
- **Feature Engineering**: Rolling statistics, momentum indicators

### Risk Management
- **Position Limits**: Maximum 30% in any single instrument
- **Transaction Costs**: Realistic cost modeling
- **Liquidity Constraints**: Based on instrument characteristics
- **Rebalancing**: Configurable frequency with drift monitoring

### Robustness
- **Multiple Methods**: Ensemble approach for hedge ratios
- **Validation**: Comprehensive data quality checks
- **Error Handling**: Graceful degradation and logging
- **Fallback Data**: Mock data when real data unavailable

## Customization Options

### Adding New Instruments
1. Edit `config/instruments.py`
2. Add instrument symbol and metadata
3. Set expected correlation with FBX
4. Specify liquidity and cost parameters

### Modifying Strategies
1. Extend `HedgeOptimizer` class
2. Add new method to `HEDGE_METHODS`
3. Implement optimization logic
4. Update reporting accordingly

### Custom Metrics
1. Add to `RiskMetrics` class
2. Include in backtesting results
3. Update Excel reporting templates

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies installed
2. **Data Issues**: Check data format and dates
3. **Memory Issues**: Reduce date range or instruments
4. **Excel Issues**: Ensure Excel libraries installed

### Performance Optimization
- **Reduce Date Range**: Shorter periods for testing
- **Limit Instruments**: Focus on high-correlation instruments
- **Parallel Processing**: For large-scale backtesting
- **Data Caching**: Save processed data for reuse

## License and Disclaimer

This system is for educational and research purposes. Real trading involves significant risk and requires careful consideration of market conditions, regulatory requirements, and risk management practices.

## Future Enhancements

1. **Real-time Data**: Live data feeds integration
2. **Machine Learning**: Advanced predictive models
3. **Web Interface**: User-friendly dashboard
4. **API Integration**: Direct broker connectivity
5. **Multi-asset**: Extend to other commodities
6. **Optimization**: Genetic algorithms for parameter tuning

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Run unit tests: `python -m pytest tests/`
3. Review configuration settings
4. Consult the detailed documentation in each module
