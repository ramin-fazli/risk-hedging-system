"""
Portfolio management module for backtesting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class Portfolio:
    """Portfolio class for managing positions and tracking performance"""
    
    def __init__(self, config, initial_capital: float = None):
        self.config = config
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.cash = self.initial_capital
        self.positions = {}  # {instrument: {'quantity': float, 'avg_price': float}}
        self.portfolio_value = self.initial_capital
        
        # Performance tracking
        self.performance_history = []
        self.position_history = []
        self.transaction_history = []
        
        # Risk management
        self.max_position_size = config.MAX_POSITION_SIZE
        self.transaction_cost = config.TRANSACTION_COST
        
    def execute_trade(self, instrument: str, quantity: float, price: float, 
                     trade_date: datetime, trade_type: str = "hedge") -> Dict[str, any]:
        """Execute a trade and update portfolio"""
        trade_result = {
            'success': False,
            'trade_date': trade_date,
            'instrument': instrument,
            'quantity': quantity,
            'price': price,
            'trade_type': trade_type,
            'transaction_cost': 0,
            'error': None
        }
        
        try:
            # Calculate transaction cost
            trade_value = abs(quantity * price)
            transaction_cost = trade_value * self.transaction_cost
            
            # Check if we have enough cash (for buys) or positions (for sells)
            if quantity > 0:  # Buy
                total_cost = trade_value + transaction_cost
                if self.cash < total_cost:
                    trade_result['error'] = f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}"
                    return trade_result
                
                # Execute buy
                self.cash -= total_cost
                self._update_position(instrument, quantity, price)
                
            else:  # Sell
                current_position = self.positions.get(instrument, {}).get('quantity', 0)
                if current_position < abs(quantity):
                    trade_result['error'] = f"Insufficient position: need {abs(quantity)}, have {current_position}"
                    return trade_result
                
                # Execute sell
                self.cash += (trade_value - transaction_cost)
                self._update_position(instrument, quantity, price)
            
            # Record transaction
            transaction_record = {
                'date': trade_date,
                'instrument': instrument,
                'quantity': quantity,
                'price': price,
                'trade_value': trade_value,
                'transaction_cost': transaction_cost,
                'trade_type': trade_type,
                'cash_after': self.cash
            }
            self.transaction_history.append(transaction_record)
            
            trade_result['success'] = True
            trade_result['transaction_cost'] = transaction_cost
            
        except Exception as e:
            trade_result['error'] = str(e)
            self.logger.error(f"Trade execution failed: {e}")
        
        return trade_result
    
    def _update_position(self, instrument: str, quantity: float, price: float):
        """Update position with new trade"""
        if instrument not in self.positions:
            self.positions[instrument] = {'quantity': 0, 'avg_price': 0}
        
        current_pos = self.positions[instrument]
        current_qty = current_pos['quantity']
        current_avg_price = current_pos['avg_price']
        
        new_quantity = current_qty + quantity
        
        if new_quantity == 0:
            # Position closed
            self.positions[instrument] = {'quantity': 0, 'avg_price': 0}
        elif (current_qty > 0 and quantity > 0) or (current_qty < 0 and quantity < 0):
            # Adding to position
            total_value = current_qty * current_avg_price + quantity * price
            new_avg_price = total_value / new_quantity if new_quantity != 0 else 0
            self.positions[instrument] = {'quantity': new_quantity, 'avg_price': new_avg_price}
        else:
            # Reducing position
            self.positions[instrument]['quantity'] = new_quantity
            # Keep the same average price for remaining position
    
    def update_portfolio_value(self, date: datetime, market_prices: Dict[str, float]):
        """Update portfolio value based on current market prices"""
        portfolio_value = self.cash
        position_values = {}
        
        for instrument, position in self.positions.items():
            quantity = position['quantity']
            if quantity != 0 and instrument in market_prices:
                current_price = market_prices[instrument]
                position_value = quantity * current_price
                portfolio_value += position_value
                position_values[instrument] = position_value
        
        self.portfolio_value = portfolio_value
        
        # Record performance
        performance_record = {
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': sum(position_values.values()),
            'position_breakdown': position_values.copy(),
            'return': (portfolio_value - self.initial_capital) / self.initial_capital,
            'daily_return': 0  # Will be calculated later
        }
        
        # Calculate daily return
        if len(self.performance_history) > 0:
            prev_value = self.performance_history[-1]['portfolio_value']
            performance_record['daily_return'] = (portfolio_value - prev_value) / prev_value if prev_value != 0 else 0
        
        self.performance_history.append(performance_record)
        
        # Record position history
        position_record = {
            'date': date,
            'positions': self.positions.copy(),
            'market_prices': market_prices.copy()
        }
        self.position_history.append(position_record)
    
    def rebalance_portfolio(self, target_positions: Dict[str, float], 
                          market_prices: Dict[str, float], 
                          rebalance_date: datetime) -> Dict[str, any]:
        """Rebalance portfolio to target positions"""
        rebalance_result = {
            'date': rebalance_date,
            'trades_executed': [],
            'rebalance_cost': 0,
            'success': True,
            'errors': []
        }
        
        try:
            # Calculate target quantities
            target_quantities = {}
            for instrument, target_weight in target_positions.items():
                if instrument in market_prices:
                    target_dollar_amount = self.portfolio_value * target_weight
                    target_quantity = target_dollar_amount / market_prices[instrument]
                    target_quantities[instrument] = target_quantity
            
            # Calculate required trades
            required_trades = {}
            for instrument, target_qty in target_quantities.items():
                current_qty = self.positions.get(instrument, {}).get('quantity', 0)
                trade_qty = target_qty - current_qty
                
                if abs(trade_qty) > 0.01:  # Only trade if meaningful difference
                    required_trades[instrument] = trade_qty
            
            # Execute trades
            for instrument, trade_qty in required_trades.items():
                if instrument in market_prices:
                    trade_result = self.execute_trade(
                        instrument, trade_qty, market_prices[instrument], 
                        rebalance_date, "rebalance"
                    )
                    
                    if trade_result['success']:
                        rebalance_result['trades_executed'].append(trade_result)
                        rebalance_result['rebalance_cost'] += trade_result['transaction_cost']
                    else:
                        rebalance_result['errors'].append(trade_result['error'])
                        rebalance_result['success'] = False
            
        except Exception as e:
            rebalance_result['success'] = False
            rebalance_result['errors'].append(str(e))
            self.logger.error(f"Rebalancing failed: {e}")
        
        return rebalance_result
    
    def calculate_portfolio_metrics(self) -> Dict[str, any]:
        """Calculate portfolio performance metrics"""
        if len(self.performance_history) == 0:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.performance_history)
        df.set_index('date', inplace=True)
        
        returns = df['daily_return'].dropna()
        
        # Basic metrics
        total_return = (df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = len(df)
        years = days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0
        
        # Transaction costs
        total_transaction_costs = sum([t['transaction_cost'] for t in self.transaction_history])
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.transaction_history),
            'total_transaction_costs': total_transaction_costs,
            'final_portfolio_value': df['portfolio_value'].iloc[-1],
            'max_portfolio_value': df['portfolio_value'].max(),
            'min_portfolio_value': df['portfolio_value'].min()
        }
        
        return metrics
    
    def get_position_summary(self) -> Dict[str, any]:
        """Get current position summary"""
        summary = {
            'total_portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'cash_percentage': self.cash / self.portfolio_value if self.portfolio_value != 0 else 0,
            'positions': {},
            'position_count': len([p for p in self.positions.values() if p['quantity'] != 0])
        }
        
        for instrument, position in self.positions.items():
            if position['quantity'] != 0:
                summary['positions'][instrument] = {
                    'quantity': position['quantity'],
                    'avg_price': position['avg_price'],
                    'market_value': 0,  # Will be updated with current prices
                    'weight': 0,
                    'unrealized_pnl': 0
                }
        
        return summary
    
    def get_performance_dataframe(self) -> pd.DataFrame:
        """Get performance history as DataFrame"""
        if len(self.performance_history) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        df.set_index('date', inplace=True)
        return df
    
    def get_transaction_dataframe(self) -> pd.DataFrame:
        """Get transaction history as DataFrame"""
        if len(self.transaction_history) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.transaction_history)
        df.set_index('date', inplace=True)
        return df
    
    def calculate_position_attribution(self, market_prices: Dict[str, float]) -> Dict[str, any]:
        """Calculate P&L attribution by position"""
        attribution = {}
        
        for instrument, position in self.positions.items():
            if position['quantity'] != 0 and instrument in market_prices:
                current_price = market_prices[instrument]
                avg_price = position['avg_price']
                quantity = position['quantity']
                
                market_value = quantity * current_price
                cost_basis = quantity * avg_price
                unrealized_pnl = market_value - cost_basis
                
                attribution[instrument] = {
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'market_value': market_value,
                    'cost_basis': cost_basis,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_return': unrealized_pnl / abs(cost_basis) if cost_basis != 0 else 0,
                    'weight': market_value / self.portfolio_value if self.portfolio_value != 0 else 0
                }
        
        return attribution
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.portfolio_value = self.initial_capital
        self.performance_history = []
        self.position_history = []
        self.transaction_history = []
