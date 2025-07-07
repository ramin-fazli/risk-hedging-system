"""
Data processing and cleaning module
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Class for processing and cleaning market data"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def process_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Main data processing pipeline"""
        processed_data = {}
        
        # Process FBX data
        self.logger.info("Processing FBX data...")
        processed_data['fbx'] = self._process_fbx_data(raw_data['fbx'])
        
        # Process instruments data
        self.logger.info("Processing instruments data...")
        processed_data['instruments'] = self._process_instruments_data(raw_data['instruments'])
        
        # Process revenue data
        self.logger.info("Processing revenue data...")
        processed_data['revenue'] = self._process_revenue_data(raw_data['revenue'])
        
        # Process market data
        self.logger.info("Processing market data...")
        processed_data['market'] = self._process_market_data(raw_data['market'])
        
        # Align all data to common dates
        self.logger.info("Aligning data to common dates...")
        processed_data = self._align_data(processed_data)
        
        # Add derived features
        self.logger.info("Adding derived features...")
        processed_data = self._add_derived_features(processed_data)
        
        return processed_data
    
    def _process_fbx_data(self, fbx_data: pd.DataFrame) -> pd.DataFrame:
        """Process FBX index data"""
        df = fbx_data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Calculate returns
        df['Returns'] = df['FBX'].pct_change()
        df['Log_Returns'] = np.log(df['FBX'] / df['FBX'].shift(1))
        
        # Calculate rolling statistics
        for window in [5, 10, 20, 60]:
            df[f'MA_{window}'] = df['FBX'].rolling(window=window).mean()
            df[f'Vol_{window}'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Calculate momentum indicators
        df['Momentum_20'] = df['FBX'] / df['FBX'].shift(20) - 1
        df['RSI'] = self._calculate_rsi(df['FBX'])
        
        # Remove extreme outliers
        df = self._remove_outliers(df, ['Returns'], method='iqr', factor=3)
        
        return df
    
    def _process_instruments_data(self, instruments_data: pd.DataFrame) -> pd.DataFrame:
        """Process hedge instruments data"""
        df = instruments_data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Calculate returns for each instrument
        returns_df = df.pct_change()
        returns_df.columns = [f"{col}_Returns" for col in returns_df.columns]
        
        # Calculate log returns
        log_returns_df = np.log(df / df.shift(1))
        log_returns_df.columns = [f"{col}_Log_Returns" for col in log_returns_df.columns]
        
        # Calculate rolling volatility
        vol_df = returns_df.rolling(window=20).std() * np.sqrt(252)
        vol_df.columns = [f"{col.replace('_Returns', '')}_Vol" for col in vol_df.columns]
        
        # Combine all data
        processed_df = pd.concat([df, returns_df, log_returns_df, vol_df], axis=1)
        
        # Remove extreme outliers
        for col in returns_df.columns:
            processed_df = self._remove_outliers(processed_df, [col], method='iqr', factor=3)
        
        return processed_df
    
    def _process_revenue_data(self, revenue_data: pd.DataFrame) -> pd.DataFrame:
        """Process revenue data"""
        df = revenue_data.copy()
        
        # Calculate revenue growth
        df['Revenue_Growth'] = df['Revenue'].pct_change()
        df['Revenue_Growth_YoY'] = df['Revenue'].pct_change(periods=4)  # Year-over-year
        
        # Calculate moving averages
        df['Revenue_MA_4Q'] = df['Revenue'].rolling(window=4).mean()
        df['Revenue_MA_8Q'] = df['Revenue'].rolling(window=8).mean()
        
        # Forward fill revenue for daily alignment
        daily_revenue = df['Revenue'].resample('D').ffill()
        daily_growth = df['Revenue_Growth'].resample('D').ffill()
        
        daily_df = pd.DataFrame({
            'Revenue': daily_revenue,
            'Revenue_Growth': daily_growth
        })
        
        return daily_df
    
    def _process_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Process market data"""
        df = market_data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Calculate real risk-free rate (assuming 2% inflation)
        df['Real_Risk_Free_Rate'] = df['risk_free_rate'] - 0.02
        
        # Calculate term structure features
        df['Rate_Change'] = df['risk_free_rate'].diff()
        df['Rate_MA_20'] = df['risk_free_rate'].rolling(window=20).mean()
        df['Rate_Volatility'] = df['Rate_Change'].rolling(window=60).std() * np.sqrt(252)
        
        return df
    
    def _align_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align all datasets to common dates"""
        # Ensure all dataframes have timezone-naive indices
        for key, df in data.items():
            if df.index.tz is not None:
                data[key] = df.tz_localize(None)
        
        # Find common date range
        all_dates = []
        for df in data.values():
            all_dates.extend(df.index.tolist())
        
        common_start = max([df.index.min() for df in data.values()])
        common_end = min([df.index.max() for df in data.values()])
        
        # Reindex all dataframes to common dates
        common_dates = pd.date_range(
            start=common_start,
            end=common_end,
            freq=self.config.FREQUENCY
        )
        
        aligned_data = {}
        for key, df in data.items():
            aligned_data[key] = df.reindex(common_dates).ffill()
        
        return aligned_data
    
    def _add_derived_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add derived features across datasets"""
        # Add interaction terms between FBX and instruments
        fbx_data = data['fbx']
        instruments_data = data['instruments']
        
        # Create correlation features
        correlation_features = pd.DataFrame(index=fbx_data.index)
        
        # Rolling correlations
        for col in instruments_data.columns:
            if col.endswith('_Returns'):
                instrument_name = col.replace('_Returns', '')
                correlation_features[f'{instrument_name}_Correlation_20'] = (
                    fbx_data['Returns'].rolling(window=20).corr(instruments_data[col])
                )
                correlation_features[f'{instrument_name}_Correlation_60'] = (
                    fbx_data['Returns'].rolling(window=60).corr(instruments_data[col])
                )
        
        # Add beta calculations
        for col in instruments_data.columns:
            if col.endswith('_Returns'):
                instrument_name = col.replace('_Returns', '')
                
                # Calculate rolling beta
                correlation_features[f'{instrument_name}_Beta_20'] = self._calculate_rolling_beta(
                    fbx_data['Returns'], instruments_data[col], window=20
                )
                correlation_features[f'{instrument_name}_Beta_60'] = self._calculate_rolling_beta(
                    fbx_data['Returns'], instruments_data[col], window=60
                )
        
        # Add to data
        data['correlations'] = correlation_features
        
        return data
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        # Forward fill first
        df = df.ffill()
        
        # Backward fill for remaining NaNs
        df = df.bfill()
        
        # Linear interpolation for any remaining gaps
        df = df.interpolate(method='linear')
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, columns: List[str], 
                        method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """Remove outliers from specified columns"""
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean.loc[z_scores > factor, col] = np.nan
                df_clean[col] = df_clean[col].fillna(method='ffill')
        
        return df_clean
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_rolling_beta(self, market_returns: pd.Series, 
                              asset_returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling beta"""
        def beta_func(x, y):
            if len(x) < 2 or len(y) < 2:
                return np.nan
            covariance = np.cov(x, y)[0, 1]
            market_variance = np.var(x)
            return covariance / market_variance if market_variance != 0 else np.nan
        
        rolling_beta = pd.Series(index=market_returns.index, dtype=float)
        
        for i in range(window, len(market_returns)):
            x = market_returns.iloc[i-window:i].values
            y = asset_returns.iloc[i-window:i].values
            
            if not (np.isnan(x).any() or np.isnan(y).any()):
                rolling_beta.iloc[i] = beta_func(x, y)
        
        return rolling_beta
    
    def generate_summary_stats(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate summary statistics for all datasets"""
        summary_stats = {}
        
        for key, df in data.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            stats_df = pd.DataFrame({
                'Mean': df[numeric_cols].mean(),
                'Std': df[numeric_cols].std(),
                'Min': df[numeric_cols].min(),
                'Max': df[numeric_cols].max(),
                'Skewness': df[numeric_cols].skew(),
                'Kurtosis': df[numeric_cols].kurtosis(),
                'Missing_Count': df[numeric_cols].isnull().sum(),
                'Missing_Percent': (df[numeric_cols].isnull().sum() / len(df)) * 100
            })
            
            summary_stats[key] = stats_df
        
        return summary_stats
