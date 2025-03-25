from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,field_validator
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
import yfinance as yf
import os
from dotenv import load_dotenv
from typing import Optional
from fastapi.responses import JSONResponse
from typing import Dict, Union
import re


load_dotenv()

app = FastAPI(title="Stock Market AI Agent API (DeepSeek)")

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

def get_stock_data(ticker: str)  -> Dict[str, Union[float, str]]:
    # Fetching real time stock price from yfinanace
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="5d")
        
        if history.empty:
            raise InvalidTickerError(f"No data found for ticker {ticker}")
            
        required_fields = ['Close', 'High', 'Low', 'Volume']
        if not all(field in history.columns for field in required_fields):
            raise DataUnavailableError("Missing essential market data fields")

        return {
            'current_price': history['Close'].iloc[-1],
            'high': history['High'].iloc[-1],
            'low': history['Low'].iloc[-1],
            'volume': history['Volume'].iloc[-1],
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A')
        }
    except ValueError as e:  # Catch general value errors
        raise InvalidTickerError(f"Invalid ticker {ticker}") from e
    except Exception as e:  # General exception catch for yfinance errors
        if "No timezone found" in str(e):
            raise DataUnavailableError("Missing timezone data")
        raise DataUnavailableError(f"YFinance error: {str(e)}")


def get_price_change(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    return {
        "5_day": hist['Close'].pct_change().iloc[-1] * 100,
        "1_month": stock.history(period="1mo")['Close'].pct_change().iloc[-1] * 100
    }

def get_volume_trend(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    avg_volume = hist['Volume'].mean()
    last_volume = hist['Volume'].iloc[-1]
    return "Above Average" if last_volume > avg_volume else "Below Average"

def get_pe_ratio(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('trailingPE', 0)
    except:
        return 0

def calculate_price_change(data: str) -> dict:
    try:
        stock = yf.Ticker(data)
        hist = stock.history(period="5d")
        return {
            "5_day": (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
        }
    except:
        return {"5_day": 0}

def analyze_volume(data: dict) -> str:
    try:
        avg = sum(data.get('volume_history', [data['volume']])) / len(data.get('volume_history', [1]))
        return "Above Average" if data['volume'] > avg else "Below Average"
    except:
        return "Unknown"

def create_stock_agent():
    """Create and configure the LangChain stock market agent"""
    
    # Set up stock data tool
    tools = [
        Tool(
            name="Stock Data",
            func=get_stock_data,
            description="Useful for fetching real-time stock prices and market data"
        )
    ]

    # Configure Model
    llm = ChatOpenAI(
        model_name="deepseek/deepseek-chat-v3-0324:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
        max_tokens=1024,
    )

    # Create agent
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="structured-chat-zero-shot-react-description",
        handle_parsing_errors=True,
        verbose=True
    )

stock_agent = create_stock_agent()

# analyze_stock endpoint
@app.post("/analyze")
async def analyze_stock(query: StockQuery):
    try:
        # Attempt to fetch stock data first
        stock_data = get_stock_data(query.ticker)
        
        # Convert dict data to formatted string for the prompt
        data_str = "\n".join([f"{k.replace('_', ' ').title()}: {v}" 
                            for k, v in stock_data.items()])
        
        prompt = f"""Analyze {query.ticker} stock using this data:
        {data_str}
        Answer: {query.question}
        Provide analysis and conclude with: 'Recommendation: [Buy/Sell/Hold]'"""
        
        response = stock_agent.invoke(prompt)
        
        # Ensure response is string
        if not isinstance(response, str):
            response = str(response)
        
        # Recommendation extraction
        recommendation_match = re.search(
            r"Recommendation:\s*(Buy|Sell|Hold)", 
            response, 
            re.IGNORECASE
        )
        recommendation = recommendation_match.group(1).capitalize() if recommendation_match else "Hold"
        print(recommendation)
        # Clean analysis text
        analysis = re.sub(
            r"\s*Recommendation:\s*(Buy|Sell|Hold).*", 
            "", 
            response, 
            flags=re.IGNORECASE | re.DOTALL
        ).strip()
        print(analysis)

        return {
            "ticker": query.ticker,
            "analysis": analysis,
            "recommendation": recommendation,
            "confidence_metrics": {
                "pe_ratio": stock_data.get('pe_ratio', 0),
                "price_change": calculate_price_change(query.ticker),
                "volume_trend": analyze_volume(stock_data)
            },
            "source": "yfinance + DeepSeek V3 (OpenRouter)"
        }

    except InvalidTickerError as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid ticker symbol: {query.ticker}", "details": str(e)}
        )
    except DataUnavailableError as e:
        return JSONResponse(
            status_code=503,
            content={"error": "Market data unavailable", "resolution": "Try again later"}
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Processing error", "message": str(e)}
        )

# endpoint to check server is up or not
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)