# Stock Market AI Agent API 

This project implements an AI agent that fetches real-time stock data and provides an analysis (including a recommendation) using LangChain and the DeepSeek API via OpenRouter. The API is built with FastAPI and leverages yfinance for market data.

## Features

- **Real-Time Stock Data:** Fetch current price, high, low, volume, P/E ratio, and market cap using yfinance.
- **AI-Powered Analysis:** Generate stock analysis and recommendations (Buy/Sell/Hold) using a LangChain agent integrated with DeepSeek.
- **REST API:** Accessible endpoints to analyze a stock and check server health.
- **Validation:** Uses Pydantic for request validation and custom error handling.

## Project Structure

- **main.py:** Entry point for the FastAPI app; defines endpoints.
- **agent.py:** Creates and configures the LangChain stock market agent.
- **models.py:** Contains Pydantic models and custom exceptions.
- **utils.py:** Utility functions for fetching stock data and computing metrics.
- **requirements.txt:** Lists required dependencies.
- **README.md:** Documentation and setup instructions.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- An API key for OpenRouter (DeepSeek). Set the environment variable:
  ```bash
  export OPENROUTER_API_KEY='your_openrouter_api_key_here'
