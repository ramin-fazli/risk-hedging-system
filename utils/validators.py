"""
Data validation utilities
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

class DataValidator:
    """Class for validating data quality and integrity"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_price_data(self, data: pd.DataFrame, 
                          required_columns: List[str] = None) -> Dict[str, Any]:
        """Validate price data"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_quality_score': 0,
            'summary': {}
        }
        
        # Basic checks
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Data is empty")
            return validation_results
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                validation_results['errors'].append(f"Missing columns: {missing_cols}")
                validation_results['is_valid'] = False
        
        # Check for negative prices
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (data[col] < 0).any():
                validation_results['warnings'].append(f"Negative values found in {col}")
        
        # Check for extreme price changes
        for col in numeric_cols:
            if col.endswith('_Returns') or 'return' in col.lower():
                extreme_returns = data[col].abs() > 0.5  # 50% daily return
                if extreme_returns.any():
                    validation_results['warnings'].append(
                        f"Extreme returns (>50%) found in {col}: {extreme_returns.sum()} observations"
                    )
        
        # Check data completeness
        missing_data = data.isnull().sum()
        completeness_score = 1 - (missing_data.sum() / (len(data) * len(data.columns)))
        
        # Check data frequency consistency
        if isinstance(data.index, pd.DatetimeIndex):
            freq_consistency = self._check_frequency_consistency(data.index)
            validation_results['summary']['frequency_consistency'] = freq_consistency
        
        # Calculate overall data quality score
        quality_score = self._calculate_quality_score(data, completeness_score)
        validation_results['data_quality_score'] = quality_score
        
        validation_results['summary'].update({
            'rows': len(data),
            'columns': len(data.columns),
            'completeness_score': completeness_score,
            'missing_values': missing_data.to_dict(),
            'date_range': (data.index.min(), data.index.max()) if isinstance(data.index, pd.DatetimeIndex) else None
        })
        
        return validation_results
    
    def validate_correlation_data(self, data: pd.DataFrame, 
                                correlation_threshold: float = 0.1) -> Dict[str, Any]:
        """Validate correlation data for hedging"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'correlation_analysis': {}
        }
        
        if len(data.columns) < 2:
            validation_results['errors'].append("Need at least 2 columns for correlation analysis")
            validation_results['is_valid'] = False
            return validation_results
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Check for high correlations (multicollinearity)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            validation_results['warnings'].append(
                f"High correlations found (>0.95): {high_corr_pairs}"
            )
        
        # Check for low correlations (poor hedging candidates)
        low_corr_instruments = []
        if len(corr_matrix.columns) > 1:
            first_col = corr_matrix.columns[0]  # Assume first column is target
            for col in corr_matrix.columns[1:]:
                if abs(corr_matrix.loc[first_col, col]) < correlation_threshold:
                    low_corr_instruments.append(col)
        
        if low_corr_instruments:
            validation_results['warnings'].append(
                f"Low correlations with target (<{correlation_threshold}): {low_corr_instruments}"
            )
        
        validation_results['correlation_analysis'] = {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'low_correlation_instruments': low_corr_instruments,
            'avg_correlation': corr_matrix.abs().mean().mean()
        }
        
        return validation_results
    
    def validate_backtest_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate data for backtesting"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_alignment': {}
        }
        
        # Check required datasets
        required_datasets = ['fbx', 'instruments', 'revenue', 'market']
        missing_datasets = set(required_datasets) - set(data.keys())
        
        if missing_datasets:
            validation_results['errors'].append(f"Missing required datasets: {missing_datasets}")
            validation_results['is_valid'] = False
        
        # Check data alignment
        date_ranges = {}
        for dataset_name, dataset in data.items():
            if isinstance(dataset.index, pd.DatetimeIndex):
                date_ranges[dataset_name] = (dataset.index.min(), dataset.index.max())
        
        if date_ranges:
            # Check overlap
            common_start = max([dr[0] for dr in date_ranges.values()])
            common_end = min([dr[1] for dr in date_ranges.values()])
            
            if common_start >= common_end:
                validation_results['errors'].append("No overlapping dates between datasets")
                validation_results['is_valid'] = False
            
            validation_results['data_alignment'] = {
                'date_ranges': date_ranges,
                'common_period': (common_start, common_end),
                'overlap_days': (common_end - common_start).days
            }
        
        # Validate each dataset
        for dataset_name, dataset in data.items():
            dataset_validation = self.validate_price_data(dataset)
            if not dataset_validation['is_valid']:
                validation_results['errors'].extend([
                    f"{dataset_name}: {error}" for error in dataset_validation['errors']
                ])
                validation_results['is_valid'] = False
            
            validation_results['warnings'].extend([
                f"{dataset_name}: {warning}" for warning in dataset_validation['warnings']
            ])
        
        return validation_results
    
    def validate_hedge_ratios(self, hedge_ratios: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hedge ratios"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'hedge_ratio_analysis': {}
        }
        
        if not hedge_ratios:
            validation_results['errors'].append("No hedge ratios provided")
            validation_results['is_valid'] = False
            return validation_results
        
        # Check for extreme hedge ratios
        extreme_ratios = {}
        for instrument, ratio_info in hedge_ratios.items():
            if isinstance(ratio_info, dict):
                ratio = ratio_info.get('hedge_ratio', 0)
                if abs(ratio) > 5:  # Extreme hedge ratio
                    extreme_ratios[instrument] = ratio
        
        if extreme_ratios:
            validation_results['warnings'].append(
                f"Extreme hedge ratios (>5): {extreme_ratios}"
            )
        
        # Check for missing hedge ratios
        zero_ratios = {}
        for instrument, ratio_info in hedge_ratios.items():
            if isinstance(ratio_info, dict):
                ratio = ratio_info.get('hedge_ratio', 0)
                if ratio == 0:
                    zero_ratios[instrument] = ratio
        
        if zero_ratios:
            validation_results['warnings'].append(
                f"Zero hedge ratios: {list(zero_ratios.keys())}"
            )
        
        # Calculate hedge ratio statistics
        ratios = []
        for instrument, ratio_info in hedge_ratios.items():
            if isinstance(ratio_info, dict):
                ratio = ratio_info.get('hedge_ratio', 0)
                ratios.append(ratio)
        
        if ratios:
            validation_results['hedge_ratio_analysis'] = {
                'mean_ratio': np.mean(ratios),
                'std_ratio': np.std(ratios),
                'min_ratio': np.min(ratios),
                'max_ratio': np.max(ratios),
                'count': len(ratios)
            }
        
        return validation_results
    
    def _check_frequency_consistency(self, date_index: pd.DatetimeIndex) -> Dict[str, Any]:
        """Check consistency of data frequency"""
        if len(date_index) < 2:
            return {'consistent': False, 'reason': 'Insufficient data'}
        
        # Calculate time differences
        time_diffs = date_index.to_series().diff().dropna()
        
        # Check if differences are consistent
        mode_diff = time_diffs.mode()[0] if not time_diffs.empty else None
        
        if mode_diff is None:
            return {'consistent': False, 'reason': 'Cannot determine frequency'}
        
        # Allow some tolerance for weekends/holidays
        tolerance = timedelta(days=3)
        inconsistent_count = (abs(time_diffs - mode_diff) > tolerance).sum()
        
        consistency_ratio = 1 - (inconsistent_count / len(time_diffs))
        
        return {
            'consistent': consistency_ratio > 0.9,
            'consistency_ratio': consistency_ratio,
            'mode_frequency': mode_diff,
            'inconsistent_observations': inconsistent_count
        }
    
    def _calculate_quality_score(self, data: pd.DataFrame, 
                               completeness_score: float) -> float:
        """Calculate overall data quality score"""
        scores = [completeness_score]
        
        # Outlier score
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_ratio = 0
        
        for col in numeric_cols:
            if len(data[col].dropna()) > 0:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)).sum()
                outlier_ratio += outliers / len(data)
        
        outlier_score = 1 - (outlier_ratio / len(numeric_cols)) if len(numeric_cols) > 0 else 1
        scores.append(outlier_score)
        
        # Consistency score (based on standard deviation)
        consistency_scores = []
        for col in numeric_cols:
            if col.endswith('_Returns') or 'return' in col.lower():
                # For return columns, check if volatility is reasonable
                vol = data[col].std()
                if 0.001 <= vol <= 0.1:  # Reasonable daily volatility range
                    consistency_scores.append(1)
                else:
                    consistency_scores.append(0.5)
        
        consistency_score = np.mean(consistency_scores) if consistency_scores else 1
        scores.append(consistency_score)
        
        return np.mean(scores)
    
    def create_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Create a formatted validation report"""
        report = []
        report.append("=== Data Validation Report ===")
        report.append(f"Overall Status: {'PASS' if validation_results['is_valid'] else 'FAIL'}")
        
        if 'data_quality_score' in validation_results:
            score = validation_results['data_quality_score']
            report.append(f"Data Quality Score: {score:.2f}/1.00")
        
        if validation_results['errors']:
            report.append("\nERRORS:")
            for error in validation_results['errors']:
                report.append(f"  - {error}")
        
        if validation_results['warnings']:
            report.append("\nWARNINGS:")
            for warning in validation_results['warnings']:
                report.append(f"  - {warning}")
        
        if 'summary' in validation_results:
            report.append("\nSUMMARY:")
            for key, value in validation_results['summary'].items():
                report.append(f"  {key}: {value}")
        
        return "\n".join(report)
