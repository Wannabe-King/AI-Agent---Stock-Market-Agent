from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models import StockQuery, InvalidTickerError, DataUnavailableError
from agent import stock_agent

from utils import get_stock_data, calculate_price_change, analyze_volume
import re



app = FastAPI(title="Stock Market AI Agent API (DeepSeek)")

# Analyze Stock Endpoint
@app.post("/analyze")
async def analyze_stock(query: StockQuery):
    """
    POST /analyze
    Fetches stock data for a given ticker, constructs a prompt with the data,
    invokes the LangChain agent to generate analysis, and returns the analysis
    along with recommendation and confidence metrics.
    """
    try:
        # Fetch real-time stock data
        stock_data = get_stock_data(query.ticker)
        
        # Format the stock data into a string for the prompt
        data_str = "\n".join([
            f"{k.replace('_', ' ').title()}: {v}" 
            for k, v in stock_data.items()
        ])
        
        # Construct the prompt for analysis
        prompt = f"""Analyze {query.ticker} stock using this data:
{data_str}
Answer: {query.question}
Provide analysis and conclude with: 'Recommendation: [Buy/Sell/Hold]'"""
        
        # Invoke the agent with the constructed prompt
        response = stock_agent.invoke(prompt)
        
        # Ensure the response is a string
        if not isinstance(response, str):
            response = str(response)
        
        # Extract recommendation using regex
        recommendation_match = re.search(
            r"Recommendation:\s*(Buy|Sell|Hold)", 
            response, 
            re.IGNORECASE
        )
        recommendation = recommendation_match.group(1).capitalize() if recommendation_match else "Hold"
        
        # Clean analysis text by removing the recommendation portion
        analysis = re.sub(
            r"\s*Recommendation:\s*(Buy|Sell|Hold).*", 
            "", 
            response, 
            flags=re.IGNORECASE | re.DOTALL
        ).strip()

        # Return the analysis, recommendation, and confidence metrics
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    GET /health
    Simple endpoint to check if the server is running.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
