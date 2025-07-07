"""
Test components of the FBX hedging system
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config
from data.mock_data import MockDataGenerator
from analysis.exposure_analyzer import ExposureAnalyzer
from analysis.hedge_optimizer import HedgeOptimizer
from utils.helpers import calculate_correlation_matrix

class TestFBXHedgingSystem(unittest.TestCase):
    """Test suite for FBX hedging system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.config.START_DATE = datetime(2023, 1, 1)
        self.config.END_DATE = datetime(2023, 12, 31)
        
        # Generate mock data
        self.mock_generator = MockDataGenerator(self.config)
        
    def test_config_validation(self):
        """Test configuration validation"""
        self.assertTrue(self.config.validate_config())
        self.assertGreater(self.config.INITIAL_CAPITAL, 0)
        self.assertLess(self.config.START_DATE, self.config.END_DATE)
    
    def test_mock_data_generation(self):
        """Test mock data generation"""
        # Test FBX data generation
        fbx_data = self.mock_generator.generate_fbx_data()
        self.assertIsInstance(fbx_data, pd.DataFrame)
        self.assertIn('FBX', fbx_data.columns)
        self.assertIn('Returns', fbx_data.columns)
        self.assertGreater(len(fbx_data), 0)
        
        # Test instrument data generation
        instrument_data = self.mock_generator.generate_instrument_data('SHIP')
        self.assertIsInstance(instrument_data, pd.Series)
        self.assertGreater(len(instrument_data), 0)
        
        # Test revenue data generation
        revenue_data = self.mock_generator.generate_revenue_data()
        self.assertIsInstance(revenue_data, pd.DataFrame)
        self.assertIn('Revenue', revenue_data.columns)
    
    def test_exposure_analyzer(self):
        """Test exposure analysis"""
        # Generate test data
        fbx_data = self.mock_generator.generate_fbx_data()
        revenue_data = self.mock_generator.generate_revenue_data()
        
        # Create exposure analyzer
        analyzer = ExposureAnalyzer(self.config)
        
        # Test FBX-Revenue relationship analysis
        relationship = analyzer._analyze_fbx_revenue_relationship(fbx_data, revenue_data)
        
        self.assertIsInstance(relationship, dict)
        self.assertIn('correlation', relationship)
        self.assertIn('linear_regression', relationship)
        
        # Check that correlation is a number
        correlation = relationship['correlation']
        self.assertIsInstance(correlation, (int, float))
        self.assertGreaterEqual(abs(correlation), 0)
        self.assertLessEqual(abs(correlation), 1)
    
    def test_hedge_optimizer(self):
        """Test hedge optimization"""
        # Generate test data
        fbx_data = self.mock_generator.generate_fbx_data()
        instruments_data = pd.DataFrame({
            'SHIP': self.mock_generator.generate_instrument_data('SHIP'),
            'IYT': self.mock_generator.generate_instrument_data('IYT')
        })
        
        # Add returns
        instruments_data['SHIP_Returns'] = instruments_data['SHIP'].pct_change()
        instruments_data['IYT_Returns'] = instruments_data['IYT'].pct_change()
        
        data = {
            'fbx': fbx_data,
            'instruments': instruments_data
        }
        
        # Create optimizer
        optimizer = HedgeOptimizer(self.config)
        
        # Test OLS hedge ratios
        hedge_ratios = optimizer._calculate_ols_hedge_ratios(data, {})
        
        self.assertIsInstance(hedge_ratios, dict)
        
        # Check that hedge ratios are calculated
        for instrument, ratio_info in hedge_ratios.items():
            self.assertIsInstance(ratio_info, dict)
            self.assertIn('hedge_ratio', ratio_info)
            self.assertIsInstance(ratio_info['hedge_ratio'], (int, float))
    
    def test_correlation_calculation(self):
        """Test correlation calculations"""
        # Generate test data
        data = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100)
        })
        
        # Calculate correlation matrix
        corr_matrix = calculate_correlation_matrix(data)
        
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (3, 3))
        
        # Check diagonal elements are 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), [1, 1, 1])
        
        # Check symmetry
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.values.T)
    
    def test_data_validation(self):
        """Test data validation"""
        # Create valid data
        valid_data = pd.DataFrame({
            'price': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Test that data is valid
        self.assertFalse(valid_data.empty)
        self.assertGreater(len(valid_data), 0)
        
        # Test missing values detection
        invalid_data = valid_data.copy()
        invalid_data.loc[2, 'price'] = np.nan
        
        missing_values = invalid_data.isnull().sum()
        self.assertGreater(missing_values['price'], 0)
    
    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        # Generate sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
        
        # Test basic metrics
        total_return = (1 + returns).prod() - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / volatility
        
        self.assertIsInstance(total_return, (int, float))
        self.assertIsInstance(volatility, (int, float))
        self.assertIsInstance(sharpe_ratio, (int, float))
        
        # Test that volatility is positive
        self.assertGreater(volatility, 0)

def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()
