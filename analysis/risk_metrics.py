"""
Risk metrics and performance analysis module
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import mean_squared_error

class RiskMetrics:
    """Class for calculating risk and performance metrics"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_portfolio_metrics(self, returns: pd.Series, 
                                  benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(returns))
        
        # Higher moment metrics
        metrics.update(self._calculate_higher_moments(returns))
        
        # Benchmark comparison (if provided)
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return-based metrics"""
        if len(returns) == 0:
            return {}
        
        # Annualized return
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252  # Assuming daily returns
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Cumulative return
        cumulative_return = total_return
        
        # Average return
        avg_return = returns.mean()
        
        # Geometric mean return
        geometric_mean = (1 + returns).prod() ** (1/len(returns)) - 1
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_return,
            'average_return': avg_return,
            'geometric_mean_return': geometric_mean
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-based metrics"""
        if len(returns) == 0:
            return {}
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Value at Risk (VaR)
        confidence_level = self.config.VAR_CONFIDENCE
        var_parametric = returns.quantile(1 - confidence_level)
        var_historical = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Conditional Value at Risk (CVaR)
        cvar = returns[returns <= var_historical].mean() if len(returns[returns <= var_historical]) > 0 else 0
        
        # Semi-variance
        semi_variance = ((returns[returns < returns.mean()] - returns.mean()) ** 2).mean()
        
        return {
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'var_parametric': var_parametric,
            'var_historical': var_historical,
            'cvar': cvar,
            'semi_variance': semi_variance
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate drawdown metrics"""
        if len(returns) == 0:
            return {}
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Drawdown duration
        drawdown_periods = self._calculate_drawdown_periods(drawdown)
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Recovery factor
        recovery_factor = returns.sum() / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'recovery_factor': recovery_factor
        }
    
    def _calculate_higher_moments(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate higher moment statistics"""
        if len(returns) == 0:
            return {}
        
        # Skewness
        skewness = returns.skew()
        
        # Kurtosis
        kurtosis = returns.kurtosis()
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
        except:
            jb_stat, jb_pvalue = np.nan, np.nan
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue
        }
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        # Align returns
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return {}
        
        portfolio_returns = aligned_data.iloc[:, 0]
        benchmark_returns = aligned_data.iloc[:, 1]
        
        # Beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Alpha
        risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate
        alpha = portfolio_returns.mean() - (risk_free_rate + beta * (benchmark_returns.mean() - risk_free_rate))
        
        # Information ratio
        excess_returns = portfolio_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Correlation
        correlation = portfolio_returns.corr(benchmark_returns)
        
        return {
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'correlation_with_benchmark': correlation
        }
    
    def _calculate_drawdown_periods(self, drawdown: pd.Series) -> List[int]:
        """Calculate drawdown periods"""
        periods = []
        current_period = 0
        in_drawdown = False
        
        for dd in drawdown:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_period = 1
                else:
                    current_period += 1
            else:
                if in_drawdown:
                    periods.append(current_period)
                    in_drawdown = False
                    current_period = 0
        
        # Handle case where series ends in drawdown
        if in_drawdown:
            periods.append(current_period)
        
        return periods
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                             risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
    
    def calculate_sortino_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() * np.sqrt(252) / downside_deviation if downside_deviation != 0 else 0
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0:
            return 0
        
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        max_drawdown = self._calculate_drawdown_metrics(returns)['max_drawdown']
        
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    def calculate_hedge_effectiveness(self, hedged_returns: pd.Series, 
                                    unhedged_returns: pd.Series) -> Dict[str, float]:
        """Calculate hedge effectiveness metrics"""
        if len(hedged_returns) == 0 or len(unhedged_returns) == 0:
            return {}
        
        # Variance reduction
        hedged_var = hedged_returns.var()
        unhedged_var = unhedged_returns.var()
        variance_reduction = (unhedged_var - hedged_var) / unhedged_var if unhedged_var != 0 else 0
        
        # Volatility reduction
        hedged_vol = hedged_returns.std()
        unhedged_vol = unhedged_returns.std()
        volatility_reduction = (unhedged_vol - hedged_vol) / unhedged_vol if unhedged_vol != 0 else 0
        
        # VaR reduction
        hedged_var_95 = hedged_returns.quantile(0.05)
        unhedged_var_95 = unhedged_returns.quantile(0.05)
        var_reduction = (unhedged_var_95 - hedged_var_95) / abs(unhedged_var_95) if unhedged_var_95 != 0 else 0
        
        # Maximum drawdown reduction
        hedged_mdd = self._calculate_drawdown_metrics(hedged_returns)['max_drawdown']
        unhedged_mdd = self._calculate_drawdown_metrics(unhedged_returns)['max_drawdown']
        mdd_reduction = (unhedged_mdd - hedged_mdd) / abs(unhedged_mdd) if unhedged_mdd != 0 else 0
        
        # Hedge ratio (regression-based)
        aligned_data = pd.concat([unhedged_returns, hedged_returns], axis=1).dropna()
        if len(aligned_data) > 1:
            hedge_effectiveness = 1 - (hedged_returns.var() / unhedged_returns.var())
        else:
            hedge_effectiveness = 0
        
        return {
            'variance_reduction': variance_reduction,
            'volatility_reduction': volatility_reduction,
            'var_reduction': var_reduction,
            'mdd_reduction': mdd_reduction,
            'hedge_effectiveness': hedge_effectiveness,
            'hedged_sharpe': self.calculate_sharpe_ratio(hedged_returns),
            'unhedged_sharpe': self.calculate_sharpe_ratio(unhedged_returns)
        }
    
    def calculate_attribution_analysis(self, portfolio_returns: pd.Series, 
                                     component_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate return attribution analysis"""
        attribution = {}
        
        total_return = portfolio_returns.sum()
        
        for component, returns in component_returns.items():
            # Align returns
            aligned_data = pd.concat([portfolio_returns, returns], axis=1).dropna()
            
            if len(aligned_data) < 2:
                continue
            
            # Calculate contribution
            component_contribution = aligned_data.iloc[:, 1].sum()
            contribution_percent = component_contribution / total_return if total_return != 0 else 0
            
            attribution[component] = {
                'total_contribution': component_contribution,
                'contribution_percent': contribution_percent,
                'average_return': aligned_data.iloc[:, 1].mean(),
                'volatility': aligned_data.iloc[:, 1].std() * np.sqrt(252)
            }
        
        return attribution
    
    def calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate tail risk metrics"""
        if len(returns) == 0:
            return {}
        
        # Expected shortfall at different confidence levels
        es_95 = returns[returns <= returns.quantile(0.05)].mean()
        es_99 = returns[returns <= returns.quantile(0.01)].mean()
        
        # Tail ratio
        tail_ratio = abs(returns.quantile(0.05)) / returns.quantile(0.95) if returns.quantile(0.95) != 0 else 0
        
        # Extreme value statistics
        extreme_positive = returns[returns > returns.quantile(0.95)].mean()
        extreme_negative = returns[returns < returns.quantile(0.05)].mean()
        
        return {
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'tail_ratio': tail_ratio,
            'extreme_positive': extreme_positive,
            'extreme_negative': extreme_negative
        }
