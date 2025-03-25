# Stock Market AI Agent API (DeepSeek)

This project implements an AI-powered stock market analysis API. It fetches real-time stock data, analyzes it using a LangChain agent integrated with DeepSeek, and provides stock recommendations.

## 🚀 Live API Endpoint
Base URL: https://ai-agent-stock-market-agent.onrender.com

Example Endpoints
Health Check: ` GET /health`

Stock Analysis: ` POST /analyze`

## Features

- **Real-Time Stock Data**: Fetches live stock prices, volume, P/E ratio, and market cap using yfinance.
- **AI-Powered Analysis**: Provides stock recommendations (Buy/Sell/Hold) using a LangChain agent.
- **REST API**: FastAPI-based endpoints for stock analysis and health check.
- **Validation**: Uses Pydantic for input validation and error handling.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- An API key for OpenRouter (DeepSeek). Set the environment variable:
  ```bash
  export OPENROUTER_API_KEY='your_openrouter_api_key_here'
  ```

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/stock_agent.git
   cd stock_agent
   ```

2. **Create and Activate a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Server:**
   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

### POST /analyze

- **Description:** Analyzes a given stock ticker and returns market insights with recommendations.
- **Request Body:**
  ```json
  {
    "ticker": "AAPL",
    "question": "What is the current stock price and recommendation?"
  }
  ```
- **Response:**
  ```json
  {
    "ticker": "AAPL",
    "analysis": "Detailed analysis text...",
    "recommendation": "Buy",
    "confidence_metrics": {
      "pe_ratio": 28.4,
      "price_change": {"5_day": 3.5},
      "volume_trend": "Above Average"
    },
    "source": "yfinance + DeepSeek V3 (OpenRouter)"
  }
  ```

### GET /health

- **Description:** Checks if the server is running.
- **Response:**
  ```json
  {
    "status": "healthy"
  }
  ```

## License

This project is open source and available under the [MIT License](LICENSE).

