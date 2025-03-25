#Contains utility functions for fetching stock data from yfinance and calculating metrics

import yfinance as yf
from typing import Dict, Union

#Fetch real-time stock data using yfinance.
def get_stock_data(ticker: str) -> Dict[str, Union[float, str]]:
    #Returns: dict: Contains current price, high, low, volume, P/E ratio, and market cap.
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="5d")
        
        if history.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        required_fields = ['Close', 'High', 'Low', 'Volume']
        if not all(field in history.columns for field in required_fields):
            raise ValueError("Missing essential market data fields")

        return {
            'current_price': history['Close'].iloc[-1],
            'high': history['High'].iloc[-1],
            'low': history['Low'].iloc[-1],
            'volume': history['Volume'].iloc[-1],
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A')
        }
    except ValueError as e:
        raise ValueError(f"Invalid ticker {ticker}") from e
    except Exception as e:
        # Specific error handling can be added here as needed.
        raise Exception(f"YFinance error: {str(e)}")

def calculate_price_change(ticker: str) -> dict:
    """
    Calculate the 5-day price change percentage for a stock.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        dict: Percentage change over 5 days.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return {
            "5_day": (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
        }
    except Exception:
        return {"5_day": 0}

def analyze_volume(data: dict) -> str:
    """
    Analyze the volume trend based on provided stock data.

    Parameters:
        data (dict): Stock data containing current volume and optionally volume history.

    Returns:
        str: "Above Average" if current volume exceeds the average; otherwise "Below Average".
    """
    try:
        volume_history = data.get('volume_history', [data['volume']])
        avg_volume = sum(volume_history) / len(volume_history) if volume_history else 1
        return "Above Average" if data.get('volume', 0) > avg_volume else "Below Average"
    except Exception:
        return "Unknown"
