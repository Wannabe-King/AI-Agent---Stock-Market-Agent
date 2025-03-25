#Defines Pydantic models for the API and custom exceptions.

from pydantic import BaseModel, field_validator
from typing import Optional

class InvalidTickerError(Exception):
    """Custom exception raised when a ticker symbol is invalid."""
    pass

class DataUnavailableError(Exception):
    """Custom exception raised when required market data is unavailable."""
    pass

class StockQuery(BaseModel):
    """
    StockQuery model for validating requests.
    
    Attributes:
        ticker: Stock ticker symbol (1-5 alphabetical characters).
        question: Analysis question (optional).
    """
    ticker: str
    question: Optional[str] = "What is the current stock price and recommendation?"

    @field_validator('ticker')
    def validate_ticker(cls, v):
        if not v.isalpha() or len(v) > 5:
            raise ValueError("Invalid ticker format. Must be 1-5 alphabetical characters")
        return v.upper()
