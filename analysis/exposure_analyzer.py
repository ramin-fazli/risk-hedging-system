"""
Exposure analysis module for quantifying revenue sensitivity to FBX
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

class ExposureAnalyzer:
    """Class for analyzing revenue exposure to FBX movements"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_exposure(self, data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Main exposure analysis pipeline"""
        results = {}
        
        # Analyze FBX-Revenue relationship
        self.logger.info("Analyzing FBX-Revenue relationship...")
        results['fbx_revenue_analysis'] = self._analyze_fbx_revenue_relationship(
            data['fbx'], data['revenue']
        )
        
        # Analyze FBX-Instruments relationship
        self.logger.info("Analyzing FBX-Instruments relationships...")
        results['fbx_instruments_analysis'] = self._analyze_fbx_instruments_relationship(
            data['fbx'], data['instruments']
        )
        
        # Calculate total exposure
        self.logger.info("Calculating total exposure...")
        results['total_exposure'] = self._calculate_total_exposure(
            data, results['fbx_revenue_analysis']
        )
        
        # Time-varying exposure analysis
        self.logger.info("Analyzing time-varying exposure...")
        results['time_varying_exposure'] = self._analyze_time_varying_exposure(
            data['fbx'], data['revenue']
        )
        
        # Cointegration analysis
        self.logger.info("Performing cointegration analysis...")
        results['cointegration_analysis'] = self._perform_cointegration_analysis(
            data['fbx'], data['instruments']
        )
        
        return results
    
    def _analyze_fbx_revenue_relationship(self, fbx_data: pd.DataFrame, 
                                        revenue_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze the relationship between FBX and revenue"""
        # Align data
        aligned_data = pd.concat([fbx_data, revenue_data], axis=1).dropna()
        
        if len(aligned_data) < 10:
            self.logger.warning("Insufficient data for FBX-Revenue analysis")
            return self._create_default_exposure_analysis()
        
        results = {}
        
        # Basic correlation analysis
        correlation = aligned_data['FBX'].corr(aligned_data['Revenue'])
        results['correlation'] = correlation
        
        # Linear regression analysis
        X = aligned_data[['FBX']].values
        y = aligned_data['Revenue'].values
        
        # OLS regression
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        results['linear_regression'] = {
            'beta': model.coef_[0],
            'intercept': model.intercept_,
            'r_squared': r2,
            'revenue_sensitivity': model.coef_[0] / aligned_data['Revenue'].mean()
        }
        
        # Statistical significance test
        X_with_const = sm.add_constant(X)
        sm_model = sm.OLS(y, X_with_const).fit()
        
        results['statistical_tests'] = {
            'beta_pvalue': sm_model.pvalues[1],
            'beta_tstat': sm_model.tvalues[1],
            'f_statistic': sm_model.fvalue,
            'f_pvalue': sm_model.f_pvalue,
            'adj_r_squared': sm_model.rsquared_adj
        }
        
        # Log-linear model (elasticity)
        if (aligned_data['FBX'] > 0).all() and (aligned_data['Revenue'] > 0).all():
            log_fbx = np.log(aligned_data['FBX'])
            log_revenue = np.log(aligned_data['Revenue'])
            
            log_model = LinearRegression()
            log_model.fit(log_fbx.values.reshape(-1, 1), log_revenue.values)
            
            results['log_linear_model'] = {
                'elasticity': log_model.coef_[0],
                'r_squared': r2_score(log_revenue.values, log_model.predict(log_fbx.values.reshape(-1, 1)))
            }
        
        # Non-linear analysis (polynomial)
        poly_features = np.column_stack([X, X**2])
        poly_model = LinearRegression()
        poly_model.fit(poly_features, y)
        
        results['polynomial_model'] = {
            'linear_coef': poly_model.coef_[0],
            'quadratic_coef': poly_model.coef_[1],
            'r_squared': r2_score(y, poly_model.predict(poly_features))
        }
        
        # Rolling correlation analysis
        rolling_corr = aligned_data['FBX'].rolling(window=20).corr(aligned_data['Revenue'])
        results['rolling_correlation'] = {
            'mean': rolling_corr.mean(),
            'std': rolling_corr.std(),
            'min': rolling_corr.min(),
            'max': rolling_corr.max(),
            'latest': rolling_corr.iloc[-1] if not rolling_corr.empty else np.nan
        }
        
        return results
    
    def _analyze_fbx_instruments_relationship(self, fbx_data: pd.DataFrame, 
                                            instruments_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze relationships between FBX and hedge instruments"""
        results = {}
        
        # Get FBX returns
        fbx_returns = fbx_data['Returns'].dropna()
        
        # Analyze each instrument
        for col in instruments_data.columns:
            if col.endswith('_Returns'):
                instrument_name = col.replace('_Returns', '')
                instrument_returns = instruments_data[col].dropna()
                
                # Align data
                aligned_data = pd.concat([fbx_returns, instrument_returns], axis=1).dropna()
                
                if len(aligned_data) < 10:
                    continue
                
                instrument_results = {}
                
                # Basic statistics
                correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                instrument_results['correlation'] = correlation
                
                # Beta calculation
                X = aligned_data.iloc[:, 0].values.reshape(-1, 1)
                y = aligned_data.iloc[:, 1].values
                
                beta_model = LinearRegression()
                beta_model.fit(X, y)
                
                instrument_results['beta'] = beta_model.coef_[0]
                instrument_results['alpha'] = beta_model.intercept_
                instrument_results['r_squared'] = r2_score(y, beta_model.predict(X))
                
                # Volatility analysis
                instrument_results['volatility'] = {
                    'fbx_vol': aligned_data.iloc[:, 0].std() * np.sqrt(252),
                    'instrument_vol': aligned_data.iloc[:, 1].std() * np.sqrt(252),
                    'vol_ratio': (aligned_data.iloc[:, 1].std() / aligned_data.iloc[:, 0].std()) 
                                if aligned_data.iloc[:, 0].std() != 0 else np.nan
                }
                
                # Tracking error
                tracking_error = (aligned_data.iloc[:, 1] - correlation * aligned_data.iloc[:, 0]).std()
                instrument_results['tracking_error'] = tracking_error * np.sqrt(252)
                
                # Downside correlation
                downside_fbx = aligned_data.iloc[:, 0] < 0
                if downside_fbx.sum() > 5:
                    downside_corr = (aligned_data.iloc[:, 0][downside_fbx]
                                   .corr(aligned_data.iloc[:, 1][downside_fbx]))
                    instrument_results['downside_correlation'] = downside_corr
                
                # Upside correlation
                upside_fbx = aligned_data.iloc[:, 0] > 0
                if upside_fbx.sum() > 5:
                    upside_corr = (aligned_data.iloc[:, 0][upside_fbx]
                                 .corr(aligned_data.iloc[:, 1][upside_fbx]))
                    instrument_results['upside_correlation'] = upside_corr
                
                results[instrument_name] = instrument_results
        
        return results
    
    def _calculate_total_exposure(self, data: Dict[str, pd.DataFrame], 
                                 fbx_revenue_analysis: Dict[str, any]) -> Dict[str, any]:
        """Calculate total exposure metrics"""
        revenue_data = data['revenue']
        fbx_data = data['fbx']
        
        # Get revenue sensitivity
        revenue_sensitivity = fbx_revenue_analysis.get('linear_regression', {}).get('revenue_sensitivity', 0)
        
        # Calculate dollar exposure
        latest_revenue = revenue_data['Revenue'].iloc[-1] if not revenue_data.empty else self.config.REVENUE_BASE
        dollar_exposure = latest_revenue * abs(revenue_sensitivity)
        
        # Calculate VaR (Value at Risk)
        fbx_vol = fbx_data['Returns'].std() * np.sqrt(252)
        confidence_level = self.config.VAR_CONFIDENCE
        var_multiplier = stats.norm.ppf(confidence_level)
        
        revenue_var = dollar_exposure * fbx_vol * var_multiplier
        
        # Calculate expected shortfall (CVaR)
        cvar = dollar_exposure * fbx_vol * (stats.norm.pdf(var_multiplier) / (1 - confidence_level))
        
        # Historical simulation VaR
        fbx_returns = fbx_data['Returns'].dropna()
        if len(fbx_returns) > 100:
            historical_var = np.percentile(fbx_returns, (1 - confidence_level) * 100)
            historical_revenue_var = dollar_exposure * abs(historical_var)
        else:
            historical_revenue_var = revenue_var
        
        exposure_results = {
            'revenue_sensitivity': revenue_sensitivity,
            'dollar_exposure': dollar_exposure,
            'fbx_volatility': fbx_vol,
            'parametric_var': revenue_var,
            'historical_var': historical_revenue_var,
            'expected_shortfall': cvar,
            'var_ratio': revenue_var / latest_revenue if latest_revenue != 0 else 0,
            'annual_revenue_at_risk': revenue_var * 4,  # Assuming quarterly revenue
            'exposure_metrics': {
                'revenue_beta': fbx_revenue_analysis.get('linear_regression', {}).get('beta', 0),
                'revenue_correlation': fbx_revenue_analysis.get('correlation', 0),
                'r_squared': fbx_revenue_analysis.get('linear_regression', {}).get('r_squared', 0)
            }
        }
        
        return exposure_results
    
    def _analyze_time_varying_exposure(self, fbx_data: pd.DataFrame, 
                                     revenue_data: pd.DataFrame) -> Dict[str, any]:
        """Analyze how exposure varies over time"""
        # Prepare data
        aligned_data = pd.concat([fbx_data, revenue_data], axis=1).dropna()
        
        if len(aligned_data) < 50:
            return {'error': 'Insufficient data for time-varying analysis'}
        
        # Rolling window analysis
        window_sizes = [20, 60, 120]  # Different time horizons
        rolling_results = {}
        
        for window in window_sizes:
            if len(aligned_data) < window:
                continue
                
            rolling_corr = aligned_data['FBX'].rolling(window=window).corr(aligned_data['Revenue'])
            rolling_beta = self._calculate_rolling_beta_exposure(
                aligned_data['FBX'], aligned_data['Revenue'], window
            )
            
            rolling_results[f'window_{window}'] = {
                'correlation': {
                    'mean': rolling_corr.mean(),
                    'std': rolling_corr.std(),
                    'trend': self._calculate_trend(rolling_corr),
                    'stability': 1 - (rolling_corr.std() / rolling_corr.mean()) if rolling_corr.mean() != 0 else 0
                },
                'beta': {
                    'mean': rolling_beta.mean(),
                    'std': rolling_beta.std(),
                    'trend': self._calculate_trend(rolling_beta),
                    'stability': 1 - (rolling_beta.std() / rolling_beta.mean()) if rolling_beta.mean() != 0 else 0
                }
            }
        
        # Regime analysis
        regime_results = self._analyze_exposure_regimes(aligned_data)
        
        return {
            'rolling_analysis': rolling_results,
            'regime_analysis': regime_results,
            'stability_score': self._calculate_stability_score(rolling_results)
        }
    
    def _perform_cointegration_analysis(self, fbx_data: pd.DataFrame, 
                                      instruments_data: pd.DataFrame) -> Dict[str, any]:
        """Perform cointegration analysis between FBX and instruments"""
        results = {}
        
        fbx_prices = fbx_data['FBX'].dropna()
        
        for col in instruments_data.columns:
            if not col.endswith('_Returns') and not col.endswith('_Vol'):
                # This is a price series
                instrument_prices = instruments_data[col].dropna()
                
                # Align data
                aligned_data = pd.concat([fbx_prices, instrument_prices], axis=1).dropna()
                
                if len(aligned_data) < 50:
                    continue
                
                try:
                    # Perform Engle-Granger cointegration test
                    coint_stat, p_value, critical_values = coint(
                        aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
                    )
                    
                    is_cointegrated = p_value < 0.05
                    
                    # Calculate cointegration relationship
                    if is_cointegrated:
                        # Estimate cointegrating vector
                        X = aligned_data.iloc[:, 0].values.reshape(-1, 1)
                        y = aligned_data.iloc[:, 1].values
                        
                        coint_model = LinearRegression()
                        coint_model.fit(X, y)
                        
                        # Calculate residuals (error correction term)
                        residuals = y - coint_model.predict(X)
                        
                        cointegration_results = {
                            'is_cointegrated': is_cointegrated,
                            'p_value': p_value,
                            'test_statistic': coint_stat,
                            'critical_values': critical_values,
                            'cointegrating_coefficient': coint_model.coef_[0],
                            'residuals_std': residuals.std(),
                            'half_life': self._calculate_half_life(residuals)
                        }
                    else:
                        cointegration_results = {
                            'is_cointegrated': is_cointegrated,
                            'p_value': p_value,
                            'test_statistic': coint_stat,
                            'critical_values': critical_values
                        }
                    
                    results[col] = cointegration_results
                    
                except Exception as e:
                    self.logger.error(f"Error in cointegration analysis for {col}: {e}")
                    continue
        
        return results
    
    def _calculate_rolling_beta_exposure(self, market: pd.Series, 
                                       asset: pd.Series, window: int) -> pd.Series:
        """Calculate rolling beta for exposure analysis"""
        rolling_beta = pd.Series(index=market.index, dtype=float)
        
        for i in range(window, len(market)):
            x = market.iloc[i-window:i].values
            y = asset.iloc[i-window:i].values
            
            if len(x) >= window and not (np.isnan(x).any() or np.isnan(y).any()):
                covariance = np.cov(x, y)[0, 1]
                variance = np.var(x)
                beta = covariance / variance if variance != 0 else np.nan
                rolling_beta.iloc[i] = beta
        
        return rolling_beta
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend in a time series"""
        if len(series.dropna()) < 2:
            return 0
        
        x = np.arange(len(series.dropna()))
        y = series.dropna().values
        
        if len(x) != len(y):
            return 0
        
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        except:
            return 0
    
    def _analyze_exposure_regimes(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze different exposure regimes"""
        # Define regimes based on FBX volatility
        fbx_vol = data['FBX'].rolling(window=20).std()
        vol_quantiles = fbx_vol.quantile([0.33, 0.67])
        
        low_vol_regime = fbx_vol <= vol_quantiles.iloc[0]
        med_vol_regime = (fbx_vol > vol_quantiles.iloc[0]) & (fbx_vol <= vol_quantiles.iloc[1])
        high_vol_regime = fbx_vol > vol_quantiles.iloc[1]
        
        regime_results = {}
        
        for regime_name, regime_mask in [
            ('low_volatility', low_vol_regime),
            ('medium_volatility', med_vol_regime),
            ('high_volatility', high_vol_regime)
        ]:
            if regime_mask.sum() > 10:
                regime_data = data[regime_mask]
                correlation = regime_data['FBX'].corr(regime_data['Revenue'])
                
                regime_results[regime_name] = {
                    'correlation': correlation,
                    'observations': regime_mask.sum(),
                    'fbx_volatility': regime_data['FBX'].std() * np.sqrt(252),
                    'revenue_volatility': regime_data['Revenue'].std() * np.sqrt(4)  # Quarterly
                }
        
        return regime_results
    
    def _calculate_stability_score(self, rolling_results: Dict[str, any]) -> float:
        """Calculate overall stability score for exposure"""
        stability_scores = []
        
        for window_results in rolling_results.values():
            if 'correlation' in window_results:
                stability_scores.append(window_results['correlation']['stability'])
            if 'beta' in window_results:
                stability_scores.append(window_results['beta']['stability'])
        
        return np.mean(stability_scores) if stability_scores else 0
    
    def _calculate_half_life(self, residuals: np.ndarray) -> float:
        """Calculate half-life of mean reversion"""
        try:
            # AR(1) model for residuals
            y = residuals[1:]
            x = residuals[:-1]
            
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)
            
            ar_coef = model.coef_[0]
            
            if ar_coef >= 1:
                return np.inf
            
            half_life = -np.log(2) / np.log(ar_coef)
            return half_life
            
        except:
            return np.nan
    
    def _create_default_exposure_analysis(self) -> Dict[str, any]:
        """Create default exposure analysis when data is insufficient"""
        return {
            'correlation': 0.0,
            'linear_regression': {
                'beta': 0.0,
                'intercept': 0.0,
                'r_squared': 0.0,
                'revenue_sensitivity': 0.0
            },
            'statistical_tests': {
                'beta_pvalue': 1.0,
                'beta_tstat': 0.0,
                'f_statistic': 0.0,
                'f_pvalue': 1.0,
                'adj_r_squared': 0.0
            },
            'rolling_correlation': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'latest': 0.0
            }
        }
