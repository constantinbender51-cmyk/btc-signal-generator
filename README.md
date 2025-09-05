# BTC Trading Signal Generator

A FastAPI application that fetches 10 years of BTC price data from Binance and uses DeepSeek AI to generate trading signals.

## Features

- Fetches 10 years of hourly BTC/USDT data from Binance
- Uses DeepSeek AI to analyze OHLC data and generate signals
- Evaluates signal profitability using future price data
- REST API endpoints for signal generation and evaluation
- Deployable on Railway

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up DeepSeek API key as environment variable:
   ```bash
   export DEEPSEEK_API_KEY='your-api-key-here'
