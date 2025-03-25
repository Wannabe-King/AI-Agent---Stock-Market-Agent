from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
import yfinance as yf
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

app = FastAPI(title="Stock Market AI Agent API (DeepSeek)")

class StockQuery(BaseModel):
    """Request model for stock price queries"""
    ticker: str
    question: Optional[str] = "What is the current stock price and recent performance?"

def get_stock_data(ticker: str) -> str:
    # Fetching real time stock price from yfinanace
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="5d")
        
        return f"""
        {ticker.upper()} Stock Data:
        - Current Price: {history['Close'].iloc[-1]:.2f}
        - 5-Day Change: {(history['Close'].iloc[-1] - history['Close'].iloc[0]):.2f}
        - Volume: {history['Volume'].iloc[-1]:,}
        - 52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}
        - Market Cap: {info.get('marketCap', 'N/A'):,}
        - PE Ratio: {info.get('trailingPE', 'N/A')}
        """
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"

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

@app.post("/analyze")
async def analyze_stock(query: StockQuery):
    try:
        prompt = f"""Analyze {query.ticker} stock using this data: 
        {get_stock_data(query.ticker)}.
        Answer: {query.question}
        Provide technical analysis with key metrics and trends."""
        
        response = stock_agent.invoke(prompt)
        return {
            "ticker": query.ticker,
            "analysis": response,
            "source": "yfinance + DeepSeek V3 (OpenRouter)"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)