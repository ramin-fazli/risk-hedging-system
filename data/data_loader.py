"""
Data loading module for fetching market data and FBX index
"""

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
from tqdm import tqdm

from config.instruments import get_all_instruments
from data.mock_data import MockDataGenerator

class DataLoader:
    """Class for loading market data from various sources"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mock_generator = MockDataGenerator(config)
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required data"""
        data = {}
        
        # Load FBX data
        self.logger.info("Loading FBX index data...")
        data['fbx'] = self.load_fbx_data()
        
        # Load hedge instruments data
        self.logger.info("Loading hedge instruments data...")
        data['instruments'] = self.load_instruments_data()
        
        # Load revenue data (mock for now)
        self.logger.info("Loading revenue data...")
        data['revenue'] = self.load_revenue_data()
        
        # Load market data (risk-free rate, etc.)
        self.logger.info("Loading market data...")
        data['market'] = self.load_market_data()
        
        return data
    
    def load_fbx_data(self) -> pd.DataFrame:
        """Load FBX index data"""
        try:
            # Try to load from CSV first
            fbx_path = self.config.get_data_path("fbx_data.csv")
            if os.path.exists(fbx_path):
                self.logger.info("Loading FBX data from CSV file")
                return pd.read_csv(fbx_path, index_col=0, parse_dates=True)
            
            # Try to fetch from external API (placeholder)
            fbx_data = self._fetch_fbx_from_api()
            if fbx_data is not None:
                return fbx_data
                
            # Generate mock data as fallback
            self.logger.warning("Using mock FBX data")
            return self.mock_generator.generate_fbx_data()
            
        except Exception as e:
            self.logger.error(f"Error loading FBX data: {e}")
            return self.mock_generator.generate_fbx_data()
    
    def load_instruments_data(self) -> pd.DataFrame:
        """Load hedge instruments price data"""
        instruments = get_all_instruments()
        data_dict = {}
        
        for symbol in tqdm(instruments, desc="Loading instruments"):
            try:
                # Try Yahoo Finance first
                if self.config.DATA_SOURCES.get("yahoo_finance", False):
                    data = self._fetch_yahoo_data(symbol)
                    if data is not None:
                        data_dict[symbol] = data['Close']
                        continue
                
                # Try CSV file
                csv_path = self.config.get_data_path(f"{symbol}_data.csv")
                if os.path.exists(csv_path):
                    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    data_dict[symbol] = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
                    continue
                
                # Generate mock data
                self.logger.warning(f"Using mock data for {symbol}")
                data_dict[symbol] = self.mock_generator.generate_instrument_data(symbol)
                
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
                data_dict[symbol] = self.mock_generator.generate_instrument_data(symbol)
        
        return pd.DataFrame(data_dict)
    
    def load_revenue_data(self) -> pd.DataFrame:
        """Load company revenue data"""
        try:
            # Try to load from CSV
            revenue_path = self.config.get_data_path("revenue_data.csv")
            if os.path.exists(revenue_path):
                return pd.read_csv(revenue_path, index_col=0, parse_dates=True)
            
            # Generate mock revenue data
            self.logger.warning("Using mock revenue data")
            return self.mock_generator.generate_revenue_data()
            
        except Exception as e:
            self.logger.error(f"Error loading revenue data: {e}")
            return self.mock_generator.generate_revenue_data()
    
    def load_market_data(self) -> pd.DataFrame:
        """Load market data (risk-free rate, etc.)"""
        try:
            # Load risk-free rate (10-year Treasury)
            treasury_data = self._fetch_yahoo_data("^TNX")
            if treasury_data is not None:
                risk_free = treasury_data['Close'] / 100  # Convert percentage to decimal
            else:
                # Mock risk-free rate
                dates = pd.date_range(
                    start=self.config.START_DATE,
                    end=self.config.END_DATE,
                    freq=self.config.FREQUENCY
                )
                risk_free = pd.Series(
                    np.random.normal(0.02, 0.005, len(dates)),
                    index=dates
                )
            
            market_data = pd.DataFrame({
                'risk_free_rate': risk_free
            })
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            return self.mock_generator.generate_market_data()
    
    def _fetch_yahoo_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                interval="1d"
            )
            
            if data.empty:
                return None
            
            # Ensure timezone-naive index
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
            return None
    
    def _fetch_fbx_from_api(self) -> Optional[pd.DataFrame]:
        """Fetch FBX data from external API (placeholder)"""
        # This is a placeholder for fetching real FBX data
        # In practice, you would integrate with Freightos API or similar
        try:
            # Example API call (not real)
            # response = requests.get("https://api.freightos.com/fbx/historical")
            # data = response.json()
            # return pd.DataFrame(data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching FBX from API: {e}")
            return None
    
    def save_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Save loaded data to CSV files"""
        try:
            os.makedirs(self.config.DATA_DIR, exist_ok=True)
            
            for key, df in data.items():
                filename = f"{key}_data.csv"
                filepath = self.config.get_data_path(filename)
                df.to_csv(filepath)
                self.logger.info(f"Saved {key} data to {filepath}")
                
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
    
    def load_cached_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load previously cached data"""
        try:
            data = {}
            required_files = ['fbx_data.csv', 'instruments_data.csv', 'revenue_data.csv', 'market_data.csv']
            
            for filename in required_files:
                filepath = self.config.get_data_path(filename)
                if not os.path.exists(filepath):
                    return None
                
                key = filename.replace('_data.csv', '')
                data[key] = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            self.logger.info("Loaded cached data successfully")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            return None
