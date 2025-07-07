"""
Version information for FBX Hedging Strategy Backtesting System
"""

__version__ = "1.0.0"
__author__ = "FBX Hedging Strategy Team"
__email__ = "contact@fbxhedging.com"
__description__ = "Advanced ML-powered hedging strategy backtesting system for FBX exposure"
__url__ = "https://github.com/yourusername/fbx-hedging-strategy"

# Version history
VERSION_HISTORY = {
    "1.0.0": {
        "date": "2025-07-07",
        "features": [
            "Complete backtesting framework",
            "Advanced ML pipeline with 11 algorithms",
            "Ensemble methods and hyperparameter optimization",
            "Professional Excel reporting",
            "Multi-configuration support",
            "Comprehensive documentation"
        ],
        "status": "Production Ready"
    }
}

def get_version():
    """Get current version"""
    return __version__

def get_info():
    """Get package information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "url": __url__
    }
