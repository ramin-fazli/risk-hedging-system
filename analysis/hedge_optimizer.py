"""
Hedge ratio optimization module
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import warnings

class HedgeOptimizer:
    """Class for optimizing hedge ratios"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def optimize_hedge_ratios(self, data: Dict[str, pd.DataFrame], 
                            exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Main hedge ratio optimization pipeline"""
        results = {}
        
        # Calculate hedge ratios using different methods
        for method in self.config.HEDGE_METHODS:
            self.logger.info(f"Calculating hedge ratios using {method} method...")
            
            if method == "ols_regression":
                results[method] = self._calculate_ols_hedge_ratios(data, exposure_results)
            elif method == "minimum_variance":
                results[method] = self._calculate_minimum_variance_hedge_ratios(data, exposure_results)
            elif method == "correlation_based":
                results[method] = self._calculate_correlation_based_hedge_ratios(data, exposure_results)
            elif method == "dynamic_beta":
                results[method] = self._calculate_dynamic_beta_hedge_ratios(data, exposure_results)
        
        # Select optimal hedge ratios
        results['optimal'] = self._select_optimal_hedge_ratios(results, data, exposure_results)
        
        # Calculate hedge effectiveness
        results['effectiveness'] = self._calculate_hedge_effectiveness(results, data, exposure_results)
        
        return results
    
    def _calculate_ols_hedge_ratios(self, data: Dict[str, pd.DataFrame], 
                                  exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Calculate hedge ratios using OLS regression"""
        fbx_returns = data['fbx']['Returns'].dropna()
        instruments_data = data['instruments']
        
        hedge_ratios = {}
        
        for col in instruments_data.columns:
            if col.endswith('_Returns'):
                instrument_name = col.replace('_Returns', '')
                instrument_returns = instruments_data[col].dropna()
                
                # Align data
                aligned_data = pd.concat([fbx_returns, instrument_returns], axis=1).dropna()
                
                if len(aligned_data) < 20:
                    continue
                
                # OLS regression
                X = aligned_data.iloc[:, 0].values.reshape(-1, 1)
                y = aligned_data.iloc[:, 1].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate metrics
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                
                hedge_ratios[instrument_name] = {
                    'hedge_ratio': model.coef_[0],
                    'alpha': model.intercept_,
                    'r_squared': 1 - (mse / np.var(y)),
                    'tracking_error': np.sqrt(mse),
                    'method': 'ols_regression'
                }
        
        return hedge_ratios
    
    def _calculate_minimum_variance_hedge_ratios(self, data: Dict[str, pd.DataFrame], 
                                               exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Calculate hedge ratios using minimum variance approach"""
        fbx_returns = data['fbx']['Returns'].dropna()
        instruments_data = data['instruments']
        
        hedge_ratios = {}
        
        for col in instruments_data.columns:
            if col.endswith('_Returns'):
                instrument_name = col.replace('_Returns', '')
                instrument_returns = instruments_data[col].dropna()
                
                # Align data
                aligned_data = pd.concat([fbx_returns, instrument_returns], axis=1).dropna()
                
                if len(aligned_data) < 20:
                    continue
                
                # Calculate covariance matrix
                cov_matrix = aligned_data.cov()
                
                # Minimum variance hedge ratio
                hedge_ratio = cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]
                
                # Calculate hedged portfolio variance
                hedged_var = (cov_matrix.iloc[0, 0] + 
                            hedge_ratio**2 * cov_matrix.iloc[1, 1] - 
                            2 * hedge_ratio * cov_matrix.iloc[0, 1])
                
                # Hedge effectiveness
                unhedged_var = cov_matrix.iloc[0, 0]
                effectiveness = 1 - (hedged_var / unhedged_var) if unhedged_var != 0 else 0
                
                hedge_ratios[instrument_name] = {
                    'hedge_ratio': hedge_ratio,
                    'hedged_variance': hedged_var,
                    'unhedged_variance': unhedged_var,
                    'hedge_effectiveness': effectiveness,
                    'method': 'minimum_variance'
                }
        
        return hedge_ratios
    
    def _calculate_correlation_based_hedge_ratios(self, data: Dict[str, pd.DataFrame], 
                                                exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Calculate hedge ratios based on correlation and volatility"""
        fbx_returns = data['fbx']['Returns'].dropna()
        instruments_data = data['instruments']
        
        hedge_ratios = {}
        
        for col in instruments_data.columns:
            if col.endswith('_Returns'):
                instrument_name = col.replace('_Returns', '')
                instrument_returns = instruments_data[col].dropna()
                
                # Align data
                aligned_data = pd.concat([fbx_returns, instrument_returns], axis=1).dropna()
                
                if len(aligned_data) < 20:
                    continue
                
                # Calculate correlation and volatilities
                correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                fbx_vol = aligned_data.iloc[:, 0].std()
                instrument_vol = aligned_data.iloc[:, 1].std()
                
                # Correlation-based hedge ratio
                hedge_ratio = correlation * (fbx_vol / instrument_vol) if instrument_vol != 0 else 0
                
                # Calculate effectiveness
                effectiveness = correlation**2
                
                hedge_ratios[instrument_name] = {
                    'hedge_ratio': hedge_ratio,
                    'correlation': correlation,
                    'fbx_volatility': fbx_vol * np.sqrt(252),
                    'instrument_volatility': instrument_vol * np.sqrt(252),
                    'hedge_effectiveness': effectiveness,
                    'method': 'correlation_based'
                }
        
        return hedge_ratios
    
    def _calculate_dynamic_beta_hedge_ratios(self, data: Dict[str, pd.DataFrame], 
                                           exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Calculate time-varying hedge ratios using dynamic beta"""
        fbx_returns = data['fbx']['Returns'].dropna()
        instruments_data = data['instruments']
        
        hedge_ratios = {}
        
        for col in instruments_data.columns:
            if col.endswith('_Returns'):
                instrument_name = col.replace('_Returns', '')
                instrument_returns = instruments_data[col].dropna()
                
                # Align data
                aligned_data = pd.concat([fbx_returns, instrument_returns], axis=1).dropna()
                
                if len(aligned_data) < 60:
                    continue
                
                # Calculate rolling beta
                window = 20
                rolling_betas = []
                
                for i in range(window, len(aligned_data)):
                    window_data = aligned_data.iloc[i-window:i]
                    
                    if len(window_data) >= window:
                        X = window_data.iloc[:, 0].values.reshape(-1, 1)
                        y = window_data.iloc[:, 1].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        rolling_betas.append(model.coef_[0])
                
                if rolling_betas:
                    # Use various statistics of rolling betas
                    current_beta = rolling_betas[-1]
                    avg_beta = np.mean(rolling_betas)
                    stable_beta = np.median(rolling_betas)
                    
                    # Beta stability measure
                    beta_stability = 1 - (np.std(rolling_betas) / np.mean(rolling_betas)) if np.mean(rolling_betas) != 0 else 0
                    
                    hedge_ratios[instrument_name] = {
                        'hedge_ratio': current_beta,
                        'average_beta': avg_beta,
                        'stable_beta': stable_beta,
                        'beta_stability': beta_stability,
                        'beta_trend': self._calculate_beta_trend(rolling_betas),
                        'method': 'dynamic_beta'
                    }
        
        return hedge_ratios
    
    def _select_optimal_hedge_ratios(self, all_results: Dict[str, Dict], 
                                   data: Dict[str, pd.DataFrame], 
                                   exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Select optimal hedge ratios based on effectiveness and stability"""
        optimal_ratios = {}
        
        # Get all instruments
        instruments = set()
        for method_results in all_results.values():
            if isinstance(method_results, dict):
                instruments.update(method_results.keys())
        
        for instrument in instruments:
            if instrument in ['effectiveness']:  # Skip meta keys
                continue
                
            instrument_results = {}
            
            # Collect results from all methods
            for method, method_results in all_results.items():
                if isinstance(method_results, dict) and instrument in method_results:
                    instrument_results[method] = method_results[instrument]
            
            if not instrument_results:
                continue
            
            # Score each method
            method_scores = {}
            
            for method, results in instrument_results.items():
                score = self._calculate_method_score(results, method)
                method_scores[method] = score
            
            # Select best method
            best_method = max(method_scores, key=method_scores.get)
            best_results = instrument_results[best_method]
            
            # Add ensemble hedge ratio (average of top methods)
            top_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            ensemble_ratio = np.mean([instrument_results[method]['hedge_ratio'] 
                                    for method, _ in top_methods])
            
            optimal_ratios[instrument] = {
                'optimal_hedge_ratio': best_results['hedge_ratio'],
                'ensemble_hedge_ratio': ensemble_ratio,
                'best_method': best_method,
                'method_scores': method_scores,
                'confidence_score': method_scores[best_method],
                'method_details': best_results
            }
        
        return optimal_ratios
    
    def _calculate_hedge_effectiveness(self, hedge_results: Dict[str, any], 
                                     data: Dict[str, pd.DataFrame], 
                                     exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Calculate hedge effectiveness for different strategies"""
        effectiveness_results = {}
        
        fbx_returns = data['fbx']['Returns'].dropna()
        instruments_data = data['instruments']
        
        # Calculate for each method
        for method in self.config.HEDGE_METHODS:
            if method not in hedge_results:
                continue
                
            method_results = hedge_results[method]
            method_effectiveness = {}
            
            for instrument, hedge_info in method_results.items():
                if isinstance(hedge_info, dict) and 'hedge_ratio' in hedge_info:
                    # Calculate effectiveness
                    effectiveness = self._calculate_individual_effectiveness(
                        instrument, hedge_info['hedge_ratio'], fbx_returns, instruments_data
                    )
                    method_effectiveness[instrument] = effectiveness
            
            effectiveness_results[method] = method_effectiveness
        
        # Calculate portfolio-level effectiveness
        effectiveness_results['portfolio'] = self._calculate_portfolio_effectiveness(
            hedge_results, fbx_returns, instruments_data, exposure_results
        )
        
        return effectiveness_results
    
    def _calculate_individual_effectiveness(self, instrument: str, hedge_ratio: float, 
                                          fbx_returns: pd.Series, 
                                          instruments_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate effectiveness for individual instrument"""
        instrument_col = f"{instrument}_Returns"
        
        if instrument_col not in instruments_data.columns:
            return {}
        
        instrument_returns = instruments_data[instrument_col].dropna()
        
        # Align data
        aligned_data = pd.concat([fbx_returns, instrument_returns], axis=1).dropna()
        
        if len(aligned_data) < 20:
            return {}
        
        # Calculate hedged returns
        hedged_returns = aligned_data.iloc[:, 0] - hedge_ratio * aligned_data.iloc[:, 1]
        
        # Effectiveness metrics
        unhedged_var = aligned_data.iloc[:, 0].var()
        hedged_var = hedged_returns.var()
        
        effectiveness = 1 - (hedged_var / unhedged_var) if unhedged_var != 0 else 0
        
        # Additional metrics
        var_reduction = (unhedged_var - hedged_var) / unhedged_var if unhedged_var != 0 else 0
        vol_reduction = (aligned_data.iloc[:, 0].std() - hedged_returns.std()) / aligned_data.iloc[:, 0].std()
        
        return {
            'hedge_effectiveness': effectiveness,
            'variance_reduction': var_reduction,
            'volatility_reduction': vol_reduction,
            'unhedged_volatility': aligned_data.iloc[:, 0].std() * np.sqrt(252),
            'hedged_volatility': hedged_returns.std() * np.sqrt(252),
            'correlation': aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
        }
    
    def _calculate_portfolio_effectiveness(self, hedge_results: Dict[str, any], 
                                         fbx_returns: pd.Series, 
                                         instruments_data: pd.DataFrame, 
                                         exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Calculate portfolio-level hedge effectiveness"""
        portfolio_results = {}
        
        for method in self.config.HEDGE_METHODS:
            if method not in hedge_results:
                continue
                
            method_results = hedge_results[method]
            
            # Create portfolio of hedge instruments
            portfolio_return = pd.Series(0, index=fbx_returns.index)
            total_weight = 0
            
            for instrument, hedge_info in method_results.items():
                if not isinstance(hedge_info, dict) or 'hedge_ratio' in hedge_info:
                    continue
                    
                instrument_col = f"{instrument}_Returns"
                if instrument_col not in instruments_data.columns:
                    continue
                
                hedge_ratio = hedge_info['hedge_ratio']
                instrument_returns = instruments_data[instrument_col].reindex(fbx_returns.index, method='ffill')
                
                # Weight by hedge effectiveness or equal weight
                weight = abs(hedge_ratio) if abs(hedge_ratio) > 0.1 else 0
                portfolio_return += weight * instrument_returns
                total_weight += weight
            
            if total_weight > 0:
                portfolio_return = portfolio_return / total_weight
                
                # Calculate portfolio hedge effectiveness
                aligned_data = pd.concat([fbx_returns, portfolio_return], axis=1).dropna()
                
                if len(aligned_data) > 20:
                    hedged_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
                    
                    unhedged_var = aligned_data.iloc[:, 0].var()
                    hedged_var = hedged_returns.var()
                    
                    effectiveness = 1 - (hedged_var / unhedged_var) if unhedged_var != 0 else 0
                    
                    portfolio_results[method] = {
                        'portfolio_effectiveness': effectiveness,
                        'portfolio_correlation': aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1]),
                        'portfolio_volatility': portfolio_return.std() * np.sqrt(252),
                        'hedged_volatility': hedged_returns.std() * np.sqrt(252),
                        'total_weight': total_weight
                    }
        
        return portfolio_results
    
    def _calculate_method_score(self, results: Dict[str, any], method: str) -> float:
        """Calculate score for a hedging method"""
        score = 0
        
        # Base score from hedge effectiveness
        if 'hedge_effectiveness' in results:
            score += results['hedge_effectiveness'] * 0.4
        elif 'r_squared' in results:
            score += results['r_squared'] * 0.4
        
        # Stability score
        if 'beta_stability' in results:
            score += results['beta_stability'] * 0.3
        elif method == 'minimum_variance':
            score += 0.2  # Minimum variance is inherently stable
        
        # Correlation strength
        if 'correlation' in results:
            score += abs(results['correlation']) * 0.3
        
        # Penalize extreme hedge ratios
        if 'hedge_ratio' in results:
            if abs(results['hedge_ratio']) > 2:
                score *= 0.8  # Penalize extreme ratios
        
        return max(0, min(1, score))  # Bound between 0 and 1
    
    def _calculate_beta_trend(self, beta_series: List[float]) -> float:
        """Calculate trend in beta series"""
        if len(beta_series) < 2:
            return 0
        
        x = np.arange(len(beta_series))
        slope, _, _, _, _ = stats.linregress(x, beta_series)
        return slope
    
    def calculate_position_sizes(self, hedge_ratios: Dict[str, any], 
                               exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Calculate optimal position sizes for hedging"""
        position_sizes = {}
        
        # Get total exposure
        total_exposure = exposure_results.get('total_exposure', {}).get('dollar_exposure', 0)
        
        if total_exposure == 0:
            return position_sizes
        
        # Get optimal hedge ratios
        optimal_ratios = hedge_ratios.get('optimal', {})
        
        for instrument, hedge_info in optimal_ratios.items():
            if not isinstance(hedge_info, dict):
                continue
                
            hedge_ratio = hedge_info.get('optimal_hedge_ratio', 0)
            confidence_score = hedge_info.get('confidence_score', 0)
            
            # Calculate position size
            base_position = total_exposure * abs(hedge_ratio)
            
            # Adjust for confidence and limits
            confidence_adjusted = base_position * confidence_score
            
            # Apply position limits
            max_position = self.config.INITIAL_CAPITAL * self.config.MAX_POSITION_SIZE
            final_position = min(confidence_adjusted, max_position)
            
            position_sizes[instrument] = {
                'dollar_amount': final_position,
                'hedge_ratio': hedge_ratio,
                'confidence_score': confidence_score,
                'position_limit_hit': confidence_adjusted > max_position,
                'exposure_covered': final_position / total_exposure if total_exposure != 0 else 0
            }
        
        return position_sizes
