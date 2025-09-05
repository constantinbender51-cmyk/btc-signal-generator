from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Dict, Any
import asyncio
from utils.data_fetcher import BTCDataFetcher
from utils.signal_evaluator import SignalEvaluator

app = FastAPI(title="BTC Trading Signal Generator", version="1.0.0")

# Global variables
btc_data = None
current_index = 0
data_fetcher = None
signal_evaluator = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global btc_data, data_fetcher, signal_evaluator
    print("Initializing BTC Signal Generator...")
    data_fetcher = BTCDataFetcher()
    signal_evaluator = SignalEvaluator()
    print("Fetching historical BTC data...")
    btc_data = data_fetcher.fetch_historical_data(years=2)  # Reduced to 2 years for faster loading
    print(f"Fetched {len(btc_data)} hourly candles")

@app.get("/")
async def root():
    return {"message": "BTC Trading Signal Generator API", "status": "active"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "data_points": len(btc_data) if btc_data is not None else 0,
        "current_index": current_index
    }

@app.get("/signal/next")
async def get_next_signal():
    """Get signal for next candle and evaluate profitability"""
    global current_index
    
    if btc_data is None or data_fetcher is None or signal_evaluator is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    if current_index + 100 >= len(btc_data):  # Need future prices for evaluation
        current_index = 0  # Reset to beginning
    
    # Get current chunk of 50 candles
    chunk = data_fetcher.get_data_chunk(btc_data, current_index, 50)
    if chunk is None:
        raise HTTPException(status_code=400, detail="Not enough data")
    
    # Format OHLC data
    ohlc_formatted = signal_evaluator.format_ohlc_data(chunk)
    
    # Generate signal using DeepSeek
    signal_data = await signal_evaluator.generate_signal(ohlc_formatted)
    
    # Get entry price (last close in the chunk)
    entry_price = float(chunk.iloc[-1]['close'])
    
    # Get future prices for evaluation (next 24 hours)
    future_start = current_index + 50
    future_end = min(future_start + 24, len(btc_data))
    future_prices = btc_data.iloc[future_start:future_end]['close'].tolist()
    
    # Evaluate profitability
    is_profitable, outcome, pnl_percent = signal_evaluator.evaluate_trade_profitability(
        signal_data['signal'], entry_price, signal_data.get('stop_price'),
        signal_data.get('target_price'), future_prices
    )
    
    # Prepare response
    response = {
        "current_index": current_index,
        "entry_timestamp": str(chunk.index[-1]),
        "entry_price": entry_price,
        "signal_data": signal_data,
        "evaluation": {
            "profitable": is_profitable,
            "outcome": outcome,
            "pnl_percent": round(pnl_percent, 2),
            "evaluation_period_hours": min(24, len(future_prices))
        },
        "next_index": current_index + 1
    }
    
    current_index += 1
    
    return JSONResponse(content=response)

@app.get("/signal/reset")
async def reset_index():
    """Reset the current index to start"""
    global current_index
    current_index = 0
    return {"message": "Index reset to 0", "current_index": current_index}

@app.get("/signal/current")
async def get_current_status():
    """Get current status and index"""
    return {
        "current_index": current_index,
        "total_candles": len(btc_data) if btc_data is not None else 0,
        "remaining_candles": len(btc_data) - current_index if btc_data is not None else 0
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
