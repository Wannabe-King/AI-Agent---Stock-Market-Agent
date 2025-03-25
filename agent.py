#The agent uses a tool to fetch real-time stock data and integrates with DeepSeek's API.

import os
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from utils import get_stock_data
from dotenv import load_dotenv

load_dotenv()

def create_stock_agent():
    """Create and configure the LangChain stock market agent."""
    
    # Define the tool for fetching stock data
    tools = [
        Tool(
            name="Stock Data",
            func=get_stock_data,
            description="Useful for fetching real-time stock prices and market data"
        )
    ]

    # Configure the DeepSeek chat model (using OpenRouter)
    llm = ChatOpenAI(
        model_name="deepseek/deepseek-chat-v3-0324:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
        max_tokens=1024,
    )

    # Initialize and return the agent with structured chat capabilities
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="structured-chat-zero-shot-react-description",
        handle_parsing_errors=True,
        verbose=True
    )

# Create the agent instance to be used by the API endpoints
stock_agent = create_stock_agent()
