"""
Main backtesting engine
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm

from backtesting.portfolio import Portfolio
from analysis.risk_metrics import RiskMetrics

class BacktestEngine:
    """Main backtesting engine for hedge strategy simulation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.risk_metrics = RiskMetrics(config)
        
    def run_backtest(self, data: Dict[str, pd.DataFrame], 
                    exposure_results: Dict[str, any], 
                    hedge_ratios: Dict[str, any]) -> Dict[str, any]:
        """Run comprehensive backtesting simulation"""
        self.logger.info("Starting backtesting simulation...")
        
        results = {}
        
        # Run unhedged simulation (baseline)
        results['unhedged'] = self._run_unhedged_simulation(data, exposure_results)
        
        # Run hedged simulations for different strategies
        for method in self.config.HEDGE_METHODS:
            if method in hedge_ratios:
                self.logger.info(f"Running hedged simulation for {method}")
                results[f'hedged_{method}'] = self._run_hedged_simulation(
                    data, exposure_results, hedge_ratios[method], method
                )
        
        # Run optimal hedged simulation
        if 'optimal' in hedge_ratios:
            self.logger.info("Running optimal hedged simulation")
            results['hedged_optimal'] = self._run_hedged_simulation(
                data, exposure_results, hedge_ratios['optimal'], 'optimal'
            )
        
        # Compare strategies
        results['comparison'] = self._compare_strategies(results)
        
        # Calculate hedge effectiveness
        results['hedge_effectiveness'] = self._calculate_hedge_effectiveness(results)
        
        # Risk analysis
        results['risk_analysis'] = self._perform_risk_analysis(results, data)
        
        # Scenario analysis
        results['scenario_analysis'] = self._perform_scenario_analysis(results, data)
        
        return results
    
    def _run_unhedged_simulation(self, data: Dict[str, pd.DataFrame], 
                               exposure_results: Dict[str, any]) -> Dict[str, any]:
        """Run unhedged (baseline) simulation"""
        fbx_data = data['fbx']
        revenue_data = data['revenue']
        
        # Calculate revenue impact from FBX movements
        revenue_sensitivity = exposure_results.get('total_exposure', {}).get('revenue_sensitivity', 0)
        
        # Simulate revenue changes
        revenue_changes = []
        portfolio_values = []
        
        initial_revenue = revenue_data['Revenue'].iloc[0] if not revenue_data.empty else self.config.REVENUE_BASE
        current_revenue = initial_revenue
        
        for i, (date, row) in enumerate(fbx_data.iterrows()):
            if i == 0:
                revenue_changes.append(0)
                portfolio_values.append(self.config.INITIAL_CAPITAL)
                continue
            
            # Calculate FBX impact on revenue
            fbx_return = row['Returns'] if not pd.isna(row['Returns']) else 0
            revenue_impact = current_revenue * revenue_sensitivity * fbx_return
            
            # Update revenue
            current_revenue += revenue_impact
            revenue_changes.append(revenue_impact)
            
            # Portfolio value (proxy for company value)
            portfolio_value = self.config.INITIAL_CAPITAL + sum(revenue_changes)
            portfolio_values.append(portfolio_value)
        
        # Calculate returns
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Calculate metrics
        metrics = self.risk_metrics.calculate_portfolio_metrics(returns)
        
        simulation_results = {
            'dates': fbx_data.index,
            'portfolio_values': portfolio_values,
            'revenue_changes': revenue_changes,
            'returns': returns,
            'metrics': metrics,
            'final_value': portfolio_values[-1] if portfolio_values else self.config.INITIAL_CAPITAL,
            'total_return': (portfolio_values[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL if portfolio_values else 0
        }
        
        return simulation_results
    
    def _run_hedged_simulation(self, data: Dict[str, pd.DataFrame], 
                             exposure_results: Dict[str, any], 
                             hedge_ratios: Dict[str, any], 
                             method: str) -> Dict[str, any]:
        """Run hedged simulation with specified hedge ratios"""
        
        # Initialize portfolio
        portfolio = Portfolio(self.config)
        
        # Get data
        fbx_data = data['fbx']
        instruments_data = data['instruments']
        revenue_data = data['revenue']
        
        # Calculate revenue sensitivity
        revenue_sensitivity = exposure_results.get('total_exposure', {}).get('revenue_sensitivity', 0)
        
        # Simulation tracking
        simulation_dates = []
        portfolio_values = []
        revenue_changes = []
        hedge_pnl = []
        total_pnl = []
        
        # Rebalancing frequency
        rebalance_freq = self.config.REBALANCE_FREQUENCY
        last_rebalance = None
        
        # Initial revenue
        initial_revenue = revenue_data['Revenue'].iloc[0] if not revenue_data.empty else self.config.REVENUE_BASE
        current_revenue = initial_revenue
        
        # Get available instruments
        available_instruments = []
        for instrument in hedge_ratios.keys():
            if isinstance(hedge_ratios[instrument], dict):
                price_col = instrument
                if price_col in instruments_data.columns:
                    available_instruments.append(instrument)
        
        # Run simulation
        for i, (date, fbx_row) in enumerate(tqdm(fbx_data.iterrows(), desc=f"Simulating {method}")):
            
            # Get market prices
            market_prices = {}
            for instrument in available_instruments:
                if instrument in instruments_data.columns:
                    price = instruments_data.loc[date, instrument] if date in instruments_data.index else np.nan
                    if not pd.isna(price):
                        market_prices[instrument] = price
            
            # Initial setup
            if i == 0:
                # Initial hedge positions
                self._establish_initial_hedge_positions(
                    portfolio, hedge_ratios, market_prices, date, exposure_results
                )
                
                portfolio.update_portfolio_value(date, market_prices)
                
                simulation_dates.append(date)
                portfolio_values.append(portfolio.portfolio_value)
                revenue_changes.append(0)
                hedge_pnl.append(0)
                total_pnl.append(0)
                
                last_rebalance = date
                continue
            
            # Calculate FBX impact on revenue
            fbx_return = fbx_row['Returns'] if not pd.isna(fbx_row['Returns']) else 0
            revenue_impact = current_revenue * revenue_sensitivity * fbx_return
            current_revenue += revenue_impact
            
            # Update portfolio with market prices
            portfolio.update_portfolio_value(date, market_prices)
            
            # Check if rebalancing is needed
            if self._should_rebalance(date, last_rebalance, rebalance_freq):
                self._rebalance_hedge_positions(
                    portfolio, hedge_ratios, market_prices, date, exposure_results
                )
                last_rebalance = date
            
            # Calculate hedge P&L
            current_hedge_pnl = portfolio.portfolio_value - self.config.INITIAL_CAPITAL
            
            # Total P&L (revenue impact + hedge P&L)
            total_current_pnl = revenue_impact + current_hedge_pnl
            
            # Track simulation
            simulation_dates.append(date)
            portfolio_values.append(portfolio.portfolio_value)
            revenue_changes.append(revenue_impact)
            hedge_pnl.append(current_hedge_pnl)
            total_pnl.append(total_current_pnl)
        
        # Calculate returns
        total_values = [self.config.INITIAL_CAPITAL + sum(revenue_changes[:i+1]) + hedge_pnl[i] 
                       for i in range(len(revenue_changes))]
        returns = pd.Series(total_values).pct_change().dropna()
        
        # Calculate metrics
        metrics = self.risk_metrics.calculate_portfolio_metrics(returns)
        portfolio_metrics = portfolio.calculate_portfolio_metrics()
        
        simulation_results = {
            'dates': simulation_dates,
            'portfolio_values': portfolio_values,
            'revenue_changes': revenue_changes,
            'hedge_pnl': hedge_pnl,
            'total_pnl': total_pnl,
            'total_values': total_values,
            'returns': returns,
            'metrics': metrics,
            'portfolio_metrics': portfolio_metrics,
            'final_value': total_values[-1] if total_values else self.config.INITIAL_CAPITAL,
            'total_return': (total_values[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL if total_values else 0,
            'hedge_return': (portfolio_values[-1] - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL if portfolio_values else 0,
            'portfolio_object': portfolio,
            'method': method
        }
        
        return simulation_results
    
    def _establish_initial_hedge_positions(self, portfolio: Portfolio, 
                                         hedge_ratios: Dict[str, any], 
                                         market_prices: Dict[str, float], 
                                         date: datetime, 
                                         exposure_results: Dict[str, any]):
        """Establish initial hedge positions"""
        total_exposure = exposure_results.get('total_exposure', {}).get('dollar_exposure', 0)
        
        if total_exposure == 0:
            return
        
        for instrument, hedge_info in hedge_ratios.items():
            if not isinstance(hedge_info, dict) or instrument not in market_prices:
                continue
            
            # Get hedge ratio
            if 'optimal_hedge_ratio' in hedge_info:
                hedge_ratio = hedge_info['optimal_hedge_ratio']
            elif 'hedge_ratio' in hedge_info:
                hedge_ratio = hedge_info['hedge_ratio']
            else:
                continue
            
            # Calculate position size
            target_exposure = total_exposure * abs(hedge_ratio) * 0.5  # 50% of exposure per instrument
            max_position = portfolio.portfolio_value * self.config.MAX_POSITION_SIZE
            
            position_size = min(target_exposure, max_position)
            quantity = position_size / market_prices[instrument]
            
            # Adjust sign for hedging
            if hedge_ratio > 0:  # Instrument moves with FBX, so we short it
                quantity = -quantity
            
            # Execute trade
            portfolio.execute_trade(instrument, quantity, market_prices[instrument], date, "initial_hedge")
    
    def _rebalance_hedge_positions(self, portfolio: Portfolio, 
                                 hedge_ratios: Dict[str, any], 
                                 market_prices: Dict[str, float], 
                                 date: datetime, 
                                 exposure_results: Dict[str, any]):
        """Rebalance hedge positions"""
        # Calculate target positions
        target_positions = {}
        
        for instrument, hedge_info in hedge_ratios.items():
            if not isinstance(hedge_info, dict) or instrument not in market_prices:
                continue
            
            # Get hedge ratio
            if 'optimal_hedge_ratio' in hedge_info:
                hedge_ratio = hedge_info['optimal_hedge_ratio']
            elif 'hedge_ratio' in hedge_info:
                hedge_ratio = hedge_info['hedge_ratio']
            else:
                continue
            
            # Calculate target weight
            target_weight = min(abs(hedge_ratio) * 0.1, self.config.MAX_POSITION_SIZE)
            target_positions[instrument] = target_weight if hedge_ratio < 0 else -target_weight
        
        # Execute rebalancing
        portfolio.rebalance_portfolio(target_positions, market_prices, date)
    
    def _should_rebalance(self, current_date: datetime, last_rebalance: datetime, 
                         frequency: str) -> bool:
        """Check if rebalancing is needed"""
        if last_rebalance is None:
            return True
        
        if frequency == "D":
            return True
        elif frequency == "W":
            return (current_date - last_rebalance).days >= 7
        elif frequency == "M":
            return (current_date - last_rebalance).days >= 30
        elif frequency == "Q":
            return (current_date - last_rebalance).days >= 90
        
        return False
    
    def _compare_strategies(self, results: Dict[str, any]) -> Dict[str, any]:
        """Compare different strategies"""
        comparison = {}
        
        # Get baseline (unhedged)
        baseline = results.get('unhedged', {})
        baseline_return = baseline.get('total_return', 0)
        baseline_vol = baseline.get('metrics', {}).get('volatility', 0)
        
        # Compare each strategy
        for strategy_name, strategy_results in results.items():
            if strategy_name == 'unhedged' or not isinstance(strategy_results, dict):
                continue
            
            strategy_return = strategy_results.get('total_return', 0)
            strategy_vol = strategy_results.get('metrics', {}).get('volatility', 0)
            
            # Calculate comparison metrics
            excess_return = strategy_return - baseline_return
            vol_reduction = (baseline_vol - strategy_vol) / baseline_vol if baseline_vol != 0 else 0
            
            # Information ratio
            if 'returns' in strategy_results and 'returns' in baseline:
                excess_returns = strategy_results['returns'] - baseline['returns']
                info_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
            else:
                info_ratio = 0
            
            comparison[strategy_name] = {
                'excess_return': excess_return,
                'volatility_reduction': vol_reduction,
                'information_ratio': info_ratio,
                'return_improvement': excess_return / abs(baseline_return) if baseline_return != 0 else 0,
                'risk_adjusted_return': strategy_return / strategy_vol if strategy_vol != 0 else 0
            }
        
        return comparison
    
    def _calculate_hedge_effectiveness(self, results: Dict[str, any]) -> Dict[str, any]:
        """Calculate hedge effectiveness metrics"""
        effectiveness = {}
        
        unhedged_results = results.get('unhedged', {})
        
        for strategy_name, strategy_results in results.items():
            if strategy_name == 'unhedged' or not isinstance(strategy_results, dict):
                continue
            
            if 'returns' in strategy_results and 'returns' in unhedged_results:
                hedge_effectiveness = self.risk_metrics.calculate_hedge_effectiveness(
                    strategy_results['returns'], unhedged_results['returns']
                )
                effectiveness[strategy_name] = hedge_effectiveness
        
        return effectiveness
    
    def _perform_risk_analysis(self, results: Dict[str, any], 
                             data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Perform comprehensive risk analysis"""
        risk_analysis = {}
        
        for strategy_name, strategy_results in results.items():
            if not isinstance(strategy_results, dict) or 'returns' not in strategy_results:
                continue
            
            returns = strategy_results['returns']
            
            # Tail risk metrics
            tail_risk = self.risk_metrics.calculate_tail_risk_metrics(returns)
            
            # Stress test scenarios
            stress_scenarios = self._stress_test_strategy(strategy_results, data)
            
            risk_analysis[strategy_name] = {
                'tail_risk': tail_risk,
                'stress_scenarios': stress_scenarios,
                'var_metrics': {
                    'var_95': returns.quantile(0.05) if len(returns) > 0 else 0,
                    'var_99': returns.quantile(0.01) if len(returns) > 0 else 0,
                    'expected_shortfall': returns[returns <= returns.quantile(0.05)].mean() if len(returns) > 0 else 0
                }
            }
        
        return risk_analysis
    
    def _stress_test_strategy(self, strategy_results: Dict[str, any], 
                            data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Perform stress testing on strategy"""
        stress_scenarios = {}
        
        if 'returns' not in strategy_results:
            return stress_scenarios
        
        returns = strategy_results['returns']
        fbx_returns = data['fbx']['Returns'].dropna()
        
        # Define stress scenarios
        scenarios = {
            'high_volatility': fbx_returns[fbx_returns.abs() > fbx_returns.std() * 2],
            'extreme_negative': fbx_returns[fbx_returns < fbx_returns.quantile(0.05)],
            'extreme_positive': fbx_returns[fbx_returns > fbx_returns.quantile(0.95)],
            'consecutive_losses': self._identify_consecutive_periods(fbx_returns, negative=True),
            'consecutive_gains': self._identify_consecutive_periods(fbx_returns, negative=False)
        }
        
        for scenario_name, scenario_periods in scenarios.items():
            if len(scenario_periods) == 0:
                continue
            
            # Get corresponding strategy returns
            scenario_returns = returns.reindex(scenario_periods).dropna()
            
            if len(scenario_returns) > 0:
                stress_scenarios[scenario_name] = {
                    'avg_return': scenario_returns.mean(),
                    'volatility': scenario_returns.std() * np.sqrt(252),
                    'worst_return': scenario_returns.min(),
                    'best_return': scenario_returns.max(),
                    'observations': len(scenario_returns)
                }
        
        return stress_scenarios
    
    def _identify_consecutive_periods(self, returns: pd.Series, negative: bool = True) -> pd.DatetimeIndex:
        """Identify consecutive periods of losses/gains"""
        if negative:
            condition = returns < 0
        else:
            condition = returns > 0
        
        # Find consecutive periods
        consecutive_periods = []
        current_streak = []
        
        for date, value in condition.items():
            if value:
                current_streak.append(date)
            else:
                if len(current_streak) >= 3:  # At least 3 consecutive periods
                    consecutive_periods.extend(current_streak)
                current_streak = []
        
        # Handle case where series ends with streak
        if len(current_streak) >= 3:
            consecutive_periods.extend(current_streak)
        
        return pd.DatetimeIndex(consecutive_periods)
    
    def _perform_scenario_analysis(self, results: Dict[str, any], 
                                 data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Perform scenario analysis"""
        scenario_analysis = {}
        
        # Define scenarios based on FBX movements
        fbx_data = data['fbx']
        
        # Market regime scenarios
        regimes = {
            'bull_market': fbx_data['Returns'] > fbx_data['Returns'].quantile(0.7),
            'bear_market': fbx_data['Returns'] < fbx_data['Returns'].quantile(0.3),
            'high_volatility': fbx_data['Returns'].rolling(20).std() > fbx_data['Returns'].rolling(20).std().quantile(0.7),
            'low_volatility': fbx_data['Returns'].rolling(20).std() < fbx_data['Returns'].rolling(20).std().quantile(0.3)
        }
        
        for strategy_name, strategy_results in results.items():
            if not isinstance(strategy_results, dict) or 'returns' not in strategy_results:
                continue
            
            strategy_scenarios = {}
            
            for regime_name, regime_mask in regimes.items():
                regime_dates = fbx_data.index[regime_mask]
                regime_returns = strategy_results['returns'].reindex(regime_dates).dropna()
                
                if len(regime_returns) > 0:
                    strategy_scenarios[regime_name] = {
                        'avg_return': regime_returns.mean(),
                        'volatility': regime_returns.std() * np.sqrt(252),
                        'sharpe_ratio': self.risk_metrics.calculate_sharpe_ratio(regime_returns),
                        'max_drawdown': self.risk_metrics.calculate_portfolio_metrics(regime_returns).get('max_drawdown', 0),
                        'observations': len(regime_returns)
                    }
            
            scenario_analysis[strategy_name] = strategy_scenarios
        
        return scenario_analysis
