"""
Mock data generator for testing and development
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from config.instruments import get_instrument_info

class MockDataGenerator:
    """Generate realistic mock data for testing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate date range
        self.dates = pd.date_range(
            start=config.START_DATE,
            end=config.END_DATE,
            freq=config.FREQUENCY
        )
        
    def generate_fbx_data(self) -> pd.DataFrame:
        """Generate mock FBX index data"""
        n_days = len(self.dates)
        
        # FBX parameters
        initial_value = self.config.FBX_BASE_VALUE
        volatility = 0.25  # 25% annual volatility
        mean_reversion = 0.05  # Mean reversion strength
        trend = 0.02  # Small upward trend
        
        # Generate correlated returns with shipping cycles
        returns = []
        current_value = initial_value
        
        for i in range(n_days):
            # Add cyclical component (seasonal shipping patterns)
            cyclical = 0.1 * np.sin(2 * np.pi * i / 365.25)
            
            # Mean reverting component
            mean_rev = -mean_reversion * (np.log(current_value) - np.log(initial_value))
            
            # Random shock
            shock = np.random.normal(0, volatility / np.sqrt(252))
            
            # Total return
            daily_return = trend / 252 + mean_rev + cyclical + shock
            returns.append(daily_return)
            
            current_value *= (1 + daily_return)
        
        # Create FBX price series
        fbx_values = [initial_value]
        for ret in returns[1:]:
            fbx_values.append(fbx_values[-1] * (1 + ret))
        
        fbx_data = pd.DataFrame({
            'FBX': fbx_values,
            'Returns': [0] + returns[1:]
        }, index=self.dates)
        
        return fbx_data
    
    def generate_instrument_data(self, symbol: str) -> pd.Series:
        """Generate mock price data for an instrument"""
        n_days = len(self.dates)
        
        # Get instrument info
        info = get_instrument_info(symbol)
        expected_correlation = info.get('expected_correlation', 0) if info else 0
        
        # Base parameters
        initial_price = np.random.uniform(20, 200)
        volatility = np.random.uniform(0.15, 0.40)  # 15-40% volatility
        
        # Generate returns with correlation to FBX
        fbx_data = self.generate_fbx_data()
        fbx_returns = fbx_data['Returns'].values
        
        # Create correlated returns
        correlation = expected_correlation
        idiosyncratic_vol = volatility * np.sqrt(1 - correlation**2)
        
        returns = []
        for i in range(n_days):
            if i == 0:
                returns.append(0)
            else:
                # Correlated component
                corr_return = correlation * fbx_returns[i]
                
                # Idiosyncratic component
                idio_return = np.random.normal(0, idiosyncratic_vol / np.sqrt(252))
                
                # Total return
                total_return = corr_return + idio_return
                returns.append(total_return)
        
        # Create price series
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.Series(prices, index=self.dates)
    
    def generate_revenue_data(self) -> pd.DataFrame:
        """Generate mock quarterly revenue data"""
        # Generate quarterly dates
        quarterly_dates = pd.date_range(
            start=self.config.START_DATE,
            end=self.config.END_DATE,
            freq='QE'
        )
        
        # Base revenue parameters
        base_revenue = self.config.REVENUE_BASE
        fbx_sensitivity = self.config.FBX_REVENUE_SENSITIVITY
        
        # Get FBX data for correlation
        fbx_data = self.generate_fbx_data()
        
        revenues = []
        for date in quarterly_dates:
            # Get FBX change for the quarter
            quarter_start = date - pd.DateOffset(months=3)
            if quarter_start in fbx_data.index and date in fbx_data.index:
                fbx_change = (fbx_data.loc[date, 'FBX'] - fbx_data.loc[quarter_start, 'FBX']) / fbx_data.loc[quarter_start, 'FBX']
            else:
                fbx_change = np.random.normal(0, 0.05)
            
            # Revenue change based on FBX sensitivity
            revenue_change = fbx_sensitivity * fbx_change + np.random.normal(0, 0.1)
            
            # Calculate revenue
            if len(revenues) == 0:
                revenue = base_revenue
            else:
                revenue = revenues[-1] * (1 + revenue_change)
            
            revenues.append(max(revenue, base_revenue * 0.5))  # Floor at 50% of base
        
        revenue_data = pd.DataFrame({
            'Revenue': revenues,
            'Quarter': [f"Q{((i % 4) + 1)}-{quarterly_dates[i].year}" for i in range(len(quarterly_dates))]
        }, index=quarterly_dates)
        
        return revenue_data
    
    def generate_market_data(self) -> pd.DataFrame:
        """Generate mock market data"""
        n_days = len(self.dates)
        
        # Risk-free rate (10-year Treasury)
        base_rate = 0.025  # 2.5% base rate
        rate_volatility = 0.01
        
        rates = []
        current_rate = base_rate
        
        for i in range(n_days):
            # Mean reverting rate
            mean_reversion = 0.1 * (base_rate - current_rate)
            shock = np.random.normal(0, rate_volatility / np.sqrt(252))
            
            rate_change = mean_reversion + shock
            current_rate += rate_change
            current_rate = max(0.001, min(current_rate, 0.10))  # Bound between 0.1% and 10%
            
            rates.append(current_rate)
        
        market_data = pd.DataFrame({
            'risk_free_rate': rates
        }, index=self.dates)
        
        return market_data
    
    def generate_economic_indicators(self) -> pd.DataFrame:
        """Generate mock economic indicators"""
        monthly_dates = pd.date_range(
            start=self.config.START_DATE,
            end=self.config.END_DATE,
            freq='M'
        )
        
        # Generate various economic indicators
        indicators = {}
        
        # Baltic Dry Index (highly correlated with FBX)
        bdi_values = []
        for i, date in enumerate(monthly_dates):
            base_value = 1500
            cyclical = 300 * np.sin(2 * np.pi * i / 12)  # Seasonal pattern
            trend = 0.02 * i  # Slight upward trend
            noise = np.random.normal(0, 100)
            
            bdi_value = base_value + cyclical + trend + noise
            bdi_values.append(max(bdi_value, 300))  # Floor at 300
        
        indicators['Baltic_Dry_Index'] = bdi_values
        
        # Global trade volume index
        trade_volume = []
        for i in range(len(monthly_dates)):
            base_volume = 100
            growth = 0.03 * i / 12  # 3% annual growth
            cyclical = 5 * np.sin(2 * np.pi * i / 12)
            shock = np.random.normal(0, 3)
            
            volume = base_volume + growth + cyclical + shock
            trade_volume.append(max(volume, 80))
        
        indicators['Global_Trade_Volume'] = trade_volume
        
        # Oil prices (affects shipping costs)
        oil_prices = []
        base_oil = 60
        for i in range(len(monthly_dates)):
            volatility_shock = np.random.normal(0, 5)
            trend = 0.01 * i  # Slight upward trend
            
            oil_price = base_oil + trend + volatility_shock
            oil_prices.append(max(oil_price, 20))
        
        indicators['Oil_Price'] = oil_prices
        
        return pd.DataFrame(indicators, index=monthly_dates)
    
    def add_realistic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add realistic features to the data"""
        # Add volume data
        if 'Close' in data.columns:
            # Generate volume based on price changes
            returns = data['Close'].pct_change().fillna(0)
            base_volume = np.random.randint(100000, 1000000)
            
            volumes = []
            for ret in returns:
                # Higher volume on larger price changes
                volume_multiplier = 1 + abs(ret) * 5
                volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)
                volumes.append(int(volume))
            
            data['Volume'] = volumes
        
        # Add bid-ask spreads
        if 'Close' in data.columns:
            spread_pct = np.random.uniform(0.001, 0.005)  # 0.1% to 0.5% spread
            data['Bid'] = data['Close'] * (1 - spread_pct/2)
            data['Ask'] = data['Close'] * (1 + spread_pct/2)
        
        return data
