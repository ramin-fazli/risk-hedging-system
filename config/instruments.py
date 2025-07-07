"""
Configuration for hedge instruments (ETFs and stocks correlated with FBX)
"""

from typing import Dict, List, Any

# Shipping and logistics related ETFs and stocks
HEDGE_INSTRUMENTS = {
    "ETFs": {
        "SHIP": {
            "name": "SPDR S&P Transportation ETF",
            "description": "Transportation sector ETF",
            "expected_correlation": -0.6,
            "liquidity": "high",
            "expense_ratio": 0.35
        },
        "IYT": {
            "name": "iShares Transportation Average ETF",
            "description": "Transportation index ETF",
            "expected_correlation": -0.5,
            "liquidity": "high",
            "expense_ratio": 0.42
        },
        "FTXD": {
            "name": "First Trust Nasdaq Transportation ETF",
            "description": "Transportation technology ETF",
            "expected_correlation": -0.4,
            "liquidity": "medium",
            "expense_ratio": 0.60
        },
        "BDRY": {
            "name": "Breakwave Dry Bulk Shipping ETF",
            "description": "Dry bulk shipping ETF",
            "expected_correlation": 0.7,  # Positive correlation with FBX
            "liquidity": "low",
            "expense_ratio": 0.75
        }
    },
    
    "STOCKS": {
        "DAC": {
            "name": "Danaos Corporation",
            "description": "Container shipping company",
            "expected_correlation": 0.6,
            "market_cap": "medium",
            "sector": "shipping"
        },
        "ZIM": {
            "name": "ZIM Integrated Shipping Services",
            "description": "Integrated shipping services",
            "expected_correlation": 0.5,
            "market_cap": "large",
            "sector": "shipping"
        },
        "MATX": {
            "name": "Matson Inc",
            "description": "Ocean transportation and logistics",
            "expected_correlation": 0.4,
            "market_cap": "medium",
            "sector": "shipping"
        },
        "STNG": {
            "name": "Scorpio Tankers Inc",
            "description": "Tanker shipping company",
            "expected_correlation": 0.3,
            "market_cap": "small",
            "sector": "shipping"
        },
        "UPS": {
            "name": "United Parcel Service",
            "description": "Package delivery and logistics",
            "expected_correlation": -0.3,
            "market_cap": "large",
            "sector": "logistics"
        },
        "FDX": {
            "name": "FedEx Corporation",
            "description": "Express transportation and logistics",
            "expected_correlation": -0.3,
            "market_cap": "large",
            "sector": "logistics"
        }
    }
}

# Alternative instruments for different strategies
ALTERNATIVE_INSTRUMENTS = {
    "COMMODITIES": {
        "BDI_FUTURES": {
            "name": "Baltic Dry Index Futures",
            "description": "Direct BDI futures contracts",
            "expected_correlation": 0.9,
            "liquidity": "medium"
        },
        "CRUDE_OIL": {
            "name": "Crude Oil Futures",
            "description": "Oil futures as shipping cost proxy",
            "expected_correlation": 0.4,
            "liquidity": "high"
        }
    },
    
    "CURRENCIES": {
        "DXY": {
            "name": "US Dollar Index",
            "description": "USD strength affects shipping costs",
            "expected_correlation": -0.2,
            "liquidity": "high"
        }
    }
}

def get_all_instruments() -> List[str]:
    """Get list of all instrument symbols"""
    instruments = []
    
    for category in HEDGE_INSTRUMENTS.values():
        instruments.extend(category.keys())
    
    return instruments

def get_instrument_info(symbol: str) -> Dict[str, Any]:
    """Get information for a specific instrument"""
    for category in HEDGE_INSTRUMENTS.values():
        if symbol in category:
            return category[symbol]
    
    return None

def get_instruments_by_correlation(min_correlation: float = 0.3) -> List[str]:
    """Get instruments with correlation above threshold"""
    instruments = []
    
    for category in HEDGE_INSTRUMENTS.values():
        for symbol, info in category.items():
            if abs(info.get("expected_correlation", 0)) >= min_correlation:
                instruments.append(symbol)
    
    return instruments

def get_high_liquidity_instruments() -> List[str]:
    """Get instruments with high liquidity"""
    instruments = []
    
    for category in HEDGE_INSTRUMENTS.values():
        for symbol, info in category.items():
            if info.get("liquidity") == "high":
                instruments.append(symbol)
    
    return instruments
