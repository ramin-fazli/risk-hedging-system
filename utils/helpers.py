"""
Utility functions and helpers
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    # Suppress some verbose loggers
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def create_directories():
    """Create necessary project directories"""
    directories = [
        'data_files',
        'reports',
        'logs',
        'charts'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amounts"""
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage values"""
    return f"{value * 100:.{decimals}f}%"

def format_number(value: float, decimals: int = 2) -> str:
    """Format numbers with proper decimals"""
    return f"{value:,.{decimals}f}"

def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for DataFrame"""
    return data.corr()

def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, 
                                window: int = 20) -> pd.Series:
    """Calculate rolling correlation between two series"""
    return series1.rolling(window=window).corr(series2)

def winsorize_series(series: pd.Series, lower_percentile: float = 0.05, 
                    upper_percentile: float = 0.95) -> pd.Series:
    """Winsorize series to remove extreme outliers"""
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    return series.clip(lower=lower_bound, upper=upper_bound)

def calculate_maximum_drawdown(returns: pd.Series) -> Dict[str, Any]:
    """Calculate maximum drawdown and related metrics"""
    if len(returns) == 0:
        return {'max_drawdown': 0, 'start_date': None, 'end_date': None, 'recovery_date': None}
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    
    max_drawdown = drawdown.min()
    max_drawdown_date = drawdown.idxmin()
    
    # Find start of drawdown period
    start_date = running_max[running_max.index <= max_drawdown_date].idxmax()
    
    # Find recovery date (if any)
    recovery_date = None
    if max_drawdown_date < drawdown.index[-1]:
        recovery_series = cumulative_returns[max_drawdown_date:] >= running_max[max_drawdown_date]
        if recovery_series.any():
            recovery_date = recovery_series[recovery_series].index[0]
    
    return {
        'max_drawdown': max_drawdown,
        'start_date': start_date,
        'end_date': max_drawdown_date,
        'recovery_date': recovery_date,
        'drawdown_duration': (max_drawdown_date - start_date).days if start_date else 0,
        'recovery_duration': (recovery_date - max_drawdown_date).days if recovery_date else None
    }

def calculate_var_es(returns: pd.Series, confidence_level: float = 0.95) -> Dict[str, float]:
    """Calculate Value at Risk and Expected Shortfall"""
    if len(returns) == 0:
        return {'var': 0, 'es': 0}
    
    var = returns.quantile(1 - confidence_level)
    es = returns[returns <= var].mean()
    
    return {
        'var': var,
        'es': es,
        'var_annualized': var * np.sqrt(252),
        'es_annualized': es * np.sqrt(252)
    }

def normalize_returns(returns: pd.Series) -> pd.Series:
    """Normalize returns to start at 1"""
    return (1 + returns).cumprod()

def calculate_information_ratio(portfolio_returns: pd.Series, 
                              benchmark_returns: pd.Series) -> float:
    """Calculate information ratio"""
    active_returns = portfolio_returns - benchmark_returns
    return active_returns.mean() / active_returns.std() if active_returns.std() != 0 else 0

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate beta coefficient"""
    aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
    
    if len(aligned_data) < 2:
        return 0
    
    covariance = aligned_data.cov().iloc[0, 1]
    market_variance = aligned_data.iloc[:, 1].var()
    
    return covariance / market_variance if market_variance != 0 else 0

def calculate_alpha(asset_returns: pd.Series, market_returns: pd.Series, 
                   risk_free_rate: float = 0.02) -> float:
    """Calculate alpha coefficient"""
    beta = calculate_beta(asset_returns, market_returns)
    
    asset_mean = asset_returns.mean() * 252
    market_mean = market_returns.mean() * 252
    
    return asset_mean - (risk_free_rate + beta * (market_mean - risk_free_rate))

def resample_returns(returns: pd.Series, frequency: str = 'M') -> pd.Series:
    """Resample returns to different frequency"""
    if frequency == 'M':
        return (1 + returns).resample('M').prod() - 1
    elif frequency == 'Q':
        return (1 + returns).resample('Q').prod() - 1
    elif frequency == 'Y':
        return (1 + returns).resample('Y').prod() - 1
    else:
        return returns

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio"""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = downside_returns.std() * np.sqrt(252)
    return excess_returns.mean() * np.sqrt(252) / downside_deviation if downside_deviation != 0 else 0

def calculate_calmar_ratio(returns: pd.Series) -> float:
    """Calculate Calmar ratio"""
    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    max_drawdown = calculate_maximum_drawdown(returns)['max_drawdown']
    
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

def create_summary_table(data: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table from dictionary of metrics"""
    summary_data = []
    
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                summary_data.append({
                    'Category': key,
                    'Metric': sub_key,
                    'Value': sub_value
                })
        else:
            summary_data.append({
                'Category': 'General',
                'Metric': key,
                'Value': value
            })
    
    return pd.DataFrame(summary_data)

def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safely divide two numbers"""
    return numerator / denominator if denominator != 0 else default

def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize returns"""
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize volatility"""
    return returns.std() * np.sqrt(periods_per_year)

def calculate_hit_ratio(returns: pd.Series) -> float:
    """Calculate hit ratio (percentage of positive returns)"""
    return (returns > 0).mean()

def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor (gross profits / gross losses)"""
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    
    return gross_profits / gross_losses if gross_losses != 0 else np.inf

def calculate_risk_parity_weights(covariance_matrix: pd.DataFrame) -> pd.Series:
    """Calculate risk parity weights"""
    try:
        inv_vol = 1 / np.sqrt(np.diag(covariance_matrix))
        weights = inv_vol / inv_vol.sum()
        return pd.Series(weights, index=covariance_matrix.index)
    except:
        # Equal weights if calculation fails
        n = len(covariance_matrix)
        return pd.Series(1/n, index=covariance_matrix.index)

def bootstrap_confidence_interval(data: pd.Series, statistic_func, 
                                confidence_level: float = 0.95, 
                                n_bootstrap: int = 1000) -> Dict[str, float]:
    """Calculate bootstrap confidence interval for a statistic"""
    if len(data) == 0:
        return {'lower': 0, 'upper': 0, 'mean': 0}
    
    bootstrap_stats = []
    n_samples = len(data)
    
    for _ in range(n_bootstrap):
        sample = data.sample(n=n_samples, replace=True)
        stat = statistic_func(sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = pd.Series(bootstrap_stats)
    alpha = 1 - confidence_level
    
    return {
        'lower': bootstrap_stats.quantile(alpha/2),
        'upper': bootstrap_stats.quantile(1 - alpha/2),
        'mean': bootstrap_stats.mean()
    }

def detect_outliers(data: pd.Series, method: str = 'iqr', 
                   factor: float = 1.5) -> pd.Series:
    """Detect outliers in a series"""
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > factor
    
    else:
        return pd.Series(False, index=data.index)

def create_performance_attribution(portfolio_returns: pd.Series, 
                                 component_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """Create performance attribution analysis"""
    attribution_data = []
    
    total_return = portfolio_returns.sum()
    
    for component_name, component_return in component_returns.items():
        aligned_data = pd.concat([portfolio_returns, component_return], axis=1).dropna()
        
        if len(aligned_data) < 2:
            continue
        
        component_contribution = aligned_data.iloc[:, 1].sum()
        contribution_pct = component_contribution / total_return if total_return != 0 else 0
        
        attribution_data.append({
            'Component': component_name,
            'Total_Return': component_return.sum(),
            'Contribution_to_Portfolio': component_contribution,
            'Contribution_Percent': contribution_pct,
            'Volatility': component_return.std() * np.sqrt(252),
            'Sharpe_Ratio': calculate_sortino_ratio(component_return)
        })
    
    return pd.DataFrame(attribution_data)

def validate_data(data: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """Validate data quality"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Check if DataFrame is empty
    if data.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.any():
        validation_results['warnings'].append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
    
    # Check for infinite values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if np.isinf(data[col]).any():
            validation_results['warnings'].append(f"Infinite values found in column: {col}")
    
    # Summary statistics
    validation_results['summary'] = {
        'rows': len(data),
        'columns': len(data.columns),
        'numeric_columns': len(numeric_columns),
        'missing_values_total': missing_values.sum(),
        'date_range': f"{data.index.min()} to {data.index.max()}" if isinstance(data.index, pd.DatetimeIndex) else None
    }
    
    return validation_results
