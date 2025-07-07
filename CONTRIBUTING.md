# Contributing to FBX Hedging Strategy Backtesting System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the FBX Hedging Strategy Backtesting System.

## 🎯 Project Overview

This is an advanced ML-powered financial analysis system for backtesting hedging strategies against the Freightos Baltic Index (FBX). The project combines traditional quantitative finance with cutting-edge machine learning techniques.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (recommended: 3.12+)
- Git
- 4-16 GB RAM (depending on ML configuration)
- Basic knowledge of Python, finance, and machine learning

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/fbx-hedging-strategy.git
   cd fbx-hedging-strategy
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run tests**
   ```bash
   python -m pytest tests/
   python system_test.py
   ```

## 📝 Contribution Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Include comprehensive docstrings for all classes and methods
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Example Code Style:
```python
def calculate_hedge_ratio(self, fbx_data: pd.Series, instrument_data: pd.Series, 
                         method: str = 'ols') -> Dict[str, float]:
    """
    Calculate hedge ratio between FBX and instrument.
    
    Args:
        fbx_data: FBX price series
        instrument_data: Instrument price series
        method: Calculation method ('ols', 'minimum_variance', etc.)
        
    Returns:
        Dictionary containing hedge ratio and statistics
        
    Raises:
        ValueError: If insufficient data or invalid method
    """
    pass
```

### Commit Messages

Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation updates
- `test:` Test additions/updates
- `refactor:` Code refactoring
- `perf:` Performance improvements

Example:
```
feat(ml): add LSTM model for time series prediction

- Implemented LSTM architecture with attention mechanism
- Added hyperparameter optimization for LSTM
- Integrated with existing ensemble framework
```

## 🧪 Testing

### Test Categories

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **System Tests**: Test complete workflows
4. **Performance Tests**: Test execution time and memory usage

### Running Tests

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# System test
python system_test.py

# Performance test
python test_performance.py
```

### Writing Tests

```python
import pytest
import pandas as pd
from analysis.exposure_analyzer import ExposureAnalyzer

class TestExposureAnalyzer:
    def setup_method(self):
        """Setup test fixtures"""
        self.config = Config()
        self.analyzer = ExposureAnalyzer(self.config)
        
    def test_calculate_exposure_valid_data(self):
        """Test exposure calculation with valid data"""
        # Arrange
        fbx_data = pd.Series([100, 105, 110, 108, 112])
        revenue_data = pd.Series([1000, 1050, 1100, 1080, 1120])
        
        # Act
        result = self.analyzer.calculate_exposure(fbx_data, revenue_data)
        
        # Assert
        assert 'beta' in result
        assert 'r_squared' in result
        assert isinstance(result['beta'], float)
```

## 🔧 Development Areas

### High Priority

1. **Additional Data Sources**: Integrate real FBX data APIs
2. **Advanced ML Models**: Transformer architectures, reinforcement learning
3. **Real-time Processing**: Streaming data capabilities
4. **Web Interface**: Dashboard for interactive analysis
5. **Performance Optimization**: Parallel processing, GPU acceleration

### Medium Priority

1. **Additional Instruments**: Cryptocurrency, commodities
2. **Risk Models**: Advanced VaR models, stress testing
3. **Reporting Enhancements**: Interactive charts, PDF reports
4. **Configuration GUI**: User-friendly configuration interface
5. **Cloud Deployment**: AWS/Azure deployment scripts

### Documentation

1. **API Documentation**: Comprehensive API docs with examples
2. **Tutorials**: Step-by-step guides for different use cases
3. **Research Papers**: Academic-style documentation
4. **Video Tutorials**: Screen recordings of system usage

## 📁 Project Structure

```
fbx-hedging-strategy/
├── config/              # Configuration management
├── data/               # Data loading and processing
├── analysis/           # Financial analysis modules
├── backtesting/        # Backtesting engine
├── ml/                 # Machine learning pipeline
├── reporting/          # Report generation
├── utils/              # Utility functions
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Usage examples
└── scripts/            # Utility scripts
```

## 🚀 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/add-new-ml-model
   ```

2. **Implement Changes**
   - Write code following style guidelines
   - Add comprehensive tests
   - Update documentation

3. **Test Changes**
   ```bash
   python -m pytest
   python system_test.py
   ```

4. **Submit Pull Request**
   - Clear title and description
   - Reference any related issues
   - Include test results
   - Add screenshots if applicable

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] System test passes
- [ ] Manual testing completed

## Screenshots
(If applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🐛 Bug Reports

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Windows 10/macOS/Linux
- Python Version: 3.12.0
- Package Version: 1.0.0
- ML Mode: minimal/testing/production

## Error Messages
```
Paste error messages here
```

## Additional Context
Any additional information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should it work?

## Alternatives Considered
Other approaches considered

## Additional Context
Any additional information
```

## 📚 Documentation Standards

### Code Documentation

- All public functions must have docstrings
- Include parameter types and descriptions
- Provide usage examples
- Document return values and exceptions

### README Updates

- Keep installation instructions current
- Update feature lists
- Include performance benchmarks
- Add new configuration options

## 🎖️ Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Academic publications (if applicable)

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private inquiries
- **Documentation**: Check existing docs first

## 🏆 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## 📈 Performance Guidelines

- Optimize for readability first, performance second
- Profile code before optimizing
- Consider memory usage in ML components
- Document performance characteristics

## 🔒 Security

- Never commit API keys or sensitive data
- Use environment variables for configuration
- Report security issues privately
- Follow secure coding practices

Thank you for contributing to the FBX Hedging Strategy Backtesting System! 🚀
