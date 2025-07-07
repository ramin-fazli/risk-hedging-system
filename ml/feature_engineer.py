"""
Advanced Feature Engineering for FBX Hedging Strategy ML Models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.signal import hilbert, find_peaks
import talib
import logging

class FeatureEngineer:
    """Advanced feature engineering for financial time series"""
    
    def __init__(self, config):
        self.config = config
        self.ml_config = config.ML_CONFIG
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.feature_names = []
        self.selected_features = []
        
    def engineer_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Engineer comprehensive features from input data
        
        Args:
            data: Dictionary containing 'fbx', 'instruments', 'market', 'revenue' data
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering...")
        
        # Combine all data into single DataFrame
        combined_data = self._combine_data(data)
        
        # Initialize feature DataFrame
        features = pd.DataFrame(index=combined_data.index)
        
        # Basic features
        features = self._add_basic_features(features, combined_data)
        
        # Technical indicators
        if self.ml_config['feature_engineering']['technical_indicators']:
            features = self._add_technical_indicators(features, combined_data)
        
        # Rolling statistics
        if self.ml_config['feature_engineering']['rolling_statistics']:
            features = self._add_rolling_statistics(features, combined_data)
        
        # Lag features
        if self.ml_config['feature_engineering']['lag_features']:
            features = self._add_lag_features(features, combined_data)
        
        # Interaction features
        if self.ml_config['feature_engineering']['interaction_features']:
            features = self._add_interaction_features(features, combined_data)
        
        # Fourier features
        if self.ml_config['feature_engineering']['fourier_features']:
            features = self._add_fourier_features(features, combined_data)
        
        # Wavelet features (if enabled)
        if self.ml_config['feature_engineering']['wavelet_features']:
            features = self._add_wavelet_features(features, combined_data)
        
        # PCA features
        if self.ml_config['feature_engineering']['pca_features']:
            features = self._add_pca_features(features, combined_data)
        
        # Polynomial features
        if self.ml_config['feature_engineering']['polynomial_features']:
            features = self._add_polynomial_features(features, combined_data)
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        self.logger.info(f"Feature engineering completed. Generated {len(features.columns)} features")
        return features
    
    def _combine_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all input data into single DataFrame"""
        combined = pd.DataFrame()
        
        # Add FBX data
        if 'fbx' in data:
            fbx_data = data['fbx'].add_prefix('fbx_')
            combined = pd.concat([combined, fbx_data], axis=1)
        
        # Add instruments data
        if 'instruments' in data:
            instruments_data = data['instruments'].add_prefix('inst_')
            combined = pd.concat([combined, instruments_data], axis=1)
        
        # Add market data
        if 'market' in data:
            market_data = data['market'].add_prefix('market_')
            combined = pd.concat([combined, market_data], axis=1)
        
        # Add revenue data (interpolated to daily)
        if 'revenue' in data:
            revenue_data = data['revenue'].resample('D').interpolate().add_prefix('rev_')
            combined = pd.concat([combined, revenue_data], axis=1)
        
        return combined.fillna(method='ffill').fillna(method='bfill')
    
    def _add_basic_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic financial features"""
        for col in data.columns:
            if col.endswith('_Close') or col.endswith('_Value'):
                # Returns
                features[f'{col}_return'] = data[col].pct_change()
                features[f'{col}_log_return'] = np.log(data[col] / data[col].shift(1))
                
                # Volatility
                features[f'{col}_volatility_20'] = features[f'{col}_return'].rolling(20).std()
                features[f'{col}_volatility_60'] = features[f'{col}_return'].rolling(60).std()
                
                # Price momentum
                features[f'{col}_momentum_5'] = data[col] / data[col].shift(5) - 1
                features[f'{col}_momentum_10'] = data[col] / data[col].shift(10) - 1
                features[f'{col}_momentum_20'] = data[col] / data[col].shift(20) - 1
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA-Lib"""
        try:
            for col in data.columns:
                if col.endswith('_Close') or col.endswith('_Value'):
                    prices = data[col].values
                    
                    # Moving averages
                    features[f'{col}_sma_20'] = talib.SMA(prices, timeperiod=20)
                    features[f'{col}_ema_20'] = talib.EMA(prices, timeperiod=20)
                    features[f'{col}_wma_20'] = talib.WMA(prices, timeperiod=20)
                    
                    # Momentum indicators
                    features[f'{col}_rsi'] = talib.RSI(prices, timeperiod=14)
                    features[f'{col}_macd'], features[f'{col}_macd_signal'], features[f'{col}_macd_hist'] = talib.MACD(prices)
                    features[f'{col}_cci'] = talib.CCI(prices, prices, prices, timeperiod=14)
                    features[f'{col}_williams_r'] = talib.WILLR(prices, prices, prices, timeperiod=14)
                    
                    # Volatility indicators
                    features[f'{col}_bbands_upper'], features[f'{col}_bbands_middle'], features[f'{col}_bbands_lower'] = talib.BBANDS(prices)
                    features[f'{col}_atr'] = talib.ATR(prices, prices, prices, timeperiod=14)
                    
                    # Volume indicators (if volume data available)
                    if f'{col.replace("_Close", "_Volume")}' in data.columns:
                        volume = data[f'{col.replace("_Close", "_Volume")}'].values
                        features[f'{col}_obv'] = talib.OBV(prices, volume)
                        features[f'{col}_ad'] = talib.AD(prices, prices, prices, volume)
                    
        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {e}")
        
        return features
    
    def _add_rolling_statistics(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features"""
        windows = [5, 10, 20, 60, 120]
        
        for col in data.columns:
            if col.endswith('_Close') or col.endswith('_Value') or col.endswith('_return'):
                for window in windows:
                    # Basic statistics
                    features[f'{col}_mean_{window}'] = data[col].rolling(window).mean()
                    features[f'{col}_std_{window}'] = data[col].rolling(window).std()
                    features[f'{col}_min_{window}'] = data[col].rolling(window).min()
                    features[f'{col}_max_{window}'] = data[col].rolling(window).max()
                    features[f'{col}_median_{window}'] = data[col].rolling(window).median()
                    
                    # Advanced statistics
                    features[f'{col}_skew_{window}'] = data[col].rolling(window).skew()
                    features[f'{col}_kurt_{window}'] = data[col].rolling(window).kurt()
                    features[f'{col}_q25_{window}'] = data[col].rolling(window).quantile(0.25)
                    features[f'{col}_q75_{window}'] = data[col].rolling(window).quantile(0.75)
                    
                    # Percentile rank
                    features[f'{col}_rank_{window}'] = data[col].rolling(window).rank(pct=True)
        
        return features
    
    def _add_lag_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        lags = [1, 2, 3, 5, 10, 20, 30, 60]
        
        for col in data.columns:
            if col.endswith('_Close') or col.endswith('_Value') or col.endswith('_return'):
                for lag in lags:
                    features[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return features
    
    def _add_interaction_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between different assets"""
        # Get return columns
        return_cols = [col for col in data.columns if col.endswith('_return')]
        
        # Pairwise correlations
        for i, col1 in enumerate(return_cols):
            for col2 in return_cols[i+1:]:
                # Rolling correlation
                features[f'{col1}_{col2}_corr_20'] = data[col1].rolling(20).corr(data[col2])
                features[f'{col1}_{col2}_corr_60'] = data[col1].rolling(60).corr(data[col2])
                
                # Beta
                features[f'{col1}_{col2}_beta_20'] = (
                    data[col1].rolling(20).cov(data[col2]) / 
                    data[col2].rolling(20).var()
                )
                
                # Ratio
                if col1.endswith('_Close') and col2.endswith('_Close'):
                    features[f'{col1}_{col2}_ratio'] = data[col1] / data[col2]
        
        return features
    
    def _add_fourier_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier transform features for cyclical patterns"""
        for col in data.columns:
            if col.endswith('_Close') or col.endswith('_Value'):
                # Remove trend
                detrended = data[col].diff().dropna()
                
                # FFT
                fft_values = np.fft.fft(detrended.values)
                freqs = np.fft.fftfreq(len(detrended))
                
                # Get dominant frequencies
                dominant_freqs = np.argsort(np.abs(fft_values))[-10:]  # Top 10 frequencies
                
                for i, freq_idx in enumerate(dominant_freqs):
                    freq = freqs[freq_idx]
                    features[f'{col}_fourier_sin_{i}'] = np.sin(2 * np.pi * freq * np.arange(len(data)))
                    features[f'{col}_fourier_cos_{i}'] = np.cos(2 * np.pi * freq * np.arange(len(data)))
        
        return features
    
    def _add_wavelet_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add wavelet transform features (if enabled)"""
        try:
            import pywt
            
            for col in data.columns:
                if col.endswith('_Close') or col.endswith('_Value'):
                    # Continuous wavelet transform
                    coeffs = pywt.wavedec(data[col].dropna().values, 'db4', level=4)
                    
                    for i, coeff in enumerate(coeffs):
                        # Resize to match original data length
                        coeff_resized = np.resize(coeff, len(data))
                        features[f'{col}_wavelet_level_{i}'] = coeff_resized
                        
        except ImportError:
            self.logger.warning("PyWavelets not installed, skipping wavelet features")
        except Exception as e:
            self.logger.warning(f"Error adding wavelet features: {e}")
        
        return features
    
    def _add_pca_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add PCA features for dimensionality reduction"""
        try:
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            numeric_data = data[numeric_cols].fillna(method='ffill').fillna(0)
            
            # Apply PCA
            pca = PCA(n_components=0.95)  # Retain 95% of variance
            pca_features = pca.fit_transform(numeric_data)
            
            # Add PCA features
            for i in range(pca_features.shape[1]):
                features[f'pca_component_{i}'] = pca_features[:, i]
                
        except Exception as e:
            self.logger.warning(f"Error adding PCA features: {e}")
        
        return features
    
    def _add_polynomial_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features for non-linear relationships"""
        # Select key features for polynomial expansion
        key_cols = [col for col in data.columns if col.endswith('_return')][:5]  # Limit to avoid explosion
        
        for col in key_cols:
            # Squared terms
            features[f'{col}_squared'] = data[col] ** 2
            
            # Interaction with other key features
            for other_col in key_cols:
                if col != other_col:
                    features[f'{col}_{other_col}_product'] = data[col] * data[other_col]
        
        return features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Forward fill, then backward fill
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining NaNs, fill with median
        for col in features.columns:
            if features[col].isna().any():
                features[col] = features[col].fillna(features[col].median())
        
        return features
    
    def select_features(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select best features using multiple methods"""
        self.logger.info("Starting feature selection...")
        
        # Remove features with low variance
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.001)
        features_selected = pd.DataFrame(
            selector.fit_transform(features),
            columns=features.columns[selector.get_support()],
            index=features.index
        )
        
        # Remove highly correlated features
        correlation_matrix = features_selected.corr().abs()
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        features_selected = features_selected.drop(columns=to_drop)
        
        # Select K best features
        max_features = min(
            self.ml_config['feature_selection']['max_features'],
            len(features_selected.columns)
        )
        
        selector = SelectKBest(score_func=f_regression, k=max_features)
        features_selected = pd.DataFrame(
            selector.fit_transform(features_selected, target),
            columns=features_selected.columns[selector.get_support()],
            index=features_selected.index
        )
        
        self.selected_features = features_selected.columns.tolist()
        self.logger.info(f"Selected {len(self.selected_features)} features")
        
        return features_selected
    
    def scale_features(self, features: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale features using specified method"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        scaled_features = pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        
        self.scalers[method] = scaler
        return scaled_features
