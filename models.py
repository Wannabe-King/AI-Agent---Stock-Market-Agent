#Defines Pydantic models for the API and custom exceptions.

from pydantic import BaseModel, field_validator
from typing import Optional

class InvalidTickerError(Exception):
    pass

class DataUnavailableError(Exception):
    pass

class StockQuery(BaseModel):
    ticker: str
    question: Optional[str] = "What is the current stock price and recommendation?"

    @field_validator('ticker')
    def validate_ticker(cls, v):
        if not v.isalpha() or len(v) > 5:
            raise ValueError("Invalid ticker format. Must be 1-5 alphabetical characters")
        return v.upper()
