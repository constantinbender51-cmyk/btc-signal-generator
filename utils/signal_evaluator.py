import json
import aiohttp
from typing import Dict, Any
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class SignalEvaluator:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY', '')
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        logger.info(f"SignalEvaluator initialized with API key: {'Yes' if self.api_key else 'No'}")
        
    def format_ohlc_data(self, df_chunk):
        """Format OHLC data for the prompt - exclude the very latest candle"""
        formatted_data = []
        for idx, row in df_chunk.iterrows():
            formatted_data.append({
                'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        return formatted_data[-11:-1]  # Get candles -11 to -2 (excluding the latest)
    
    async def generate_signal(self, ohlc_data: list) -> Dict[str, Any]:
        """Generate trading signal using DeepSeek API or fallback"""
        
        # If no API key, use fallback
        if not self.api_key:
            logger.warning("No DeepSeek API key found, using fallback signal generator")
            return self._generate_fallback_signal(ohlc_data)
        
        prompt = f"""
        Analyze the following BTC/USDT hourly OHLC data and generate a trading signal.
        Respond with ONLY a JSON object containing: signal (BUY|SELL|HOLD), stop_price, target_price, 
        confidence (0-100), and reason.
        
        OHLC Data (most recent last):
        {json.dumps(ohlc_data, indent=2)}
        
        Important: Consider technical analysis, price action, volume patterns, and market structure.
        Provide realistic stop and target prices based on support/resistance levels.
        Return ONLY JSON, no other text.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert cryptocurrency trading analyst. Analyze OHLC data and provide clear trading signals with proper risk management. Always respond with ONLY a valid JSON object."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_content = result['choices'][0]['message']['content']
                        
                        # Extract JSON from response
                        try:
                            # Clean the response to extract JSON
                            json_str = response_content.strip()
                            if '```json' in json_str:
                                json_str = json_str.split('```json')[1].split('```')[0].strip()
                            elif '```' in json_str:
                                json_str = json_str.split('```')[1].split('```')[0].strip()
                                
                            signal_data = json.loads(json_str)
                            return signal_data
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON response: {e}")
                            logger.error(f"Response content: {response_content}")
                            return self._generate_fallback_signal(ohlc_data)
                    
                    logger.error(f"API request failed with status {response.status}")
                    return self._generate_fallback_signal(ohlc_data)
                    
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return self._generate_fallback_signal(ohlc_data)
    
    def _generate_fallback_signal(self, ohlc_data):
        """Generate a fallback signal without API - based on previous candle"""
        # ohlc_data now contains candles up to the previous one (last item is previous candle)
        latest = ohlc_data[-1]  # This is now the previous candle
        prev = ohlc_data[-2] if len(ohlc_data) > 1 else latest
        
        # Simple trend detection based on previous candle
        if latest['close'] > latest['open'] and latest['close'] > prev['close']:
            signal = "BUY"
            confidence = 65
            reason = "Bullish candle with upward trend"
        elif latest['close'] < latest['open'] and latest['close'] < prev['close']:
            signal = "SELL"
            confidence = 65
            reason = "Bearish candle with downward trend"
        else:
            signal = "HOLD"
            confidence = 50
            reason = "Sideways movement, no clear trend"
        
        # Calculate stop and target based on previous candle
        if signal != "HOLD":
            atr = (latest['high'] - latest['low']) * 0.5
            if signal == "BUY":
                stop_price = latest['low'] - atr
                target_price = latest['close'] + (2 * atr)
            else:
                stop_price = latest['high'] + atr
                target_price = latest['close'] - (2 * atr)
        else:
            stop_price = None
            target_price = None
        
        return {
            "signal": signal,
            "stop_price": round(stop_price, 2) if stop_price else None,
            "target_price": round(target_price, 2) if target_price else None,
            "confidence": confidence,
            "reason": f"Fallback signal: {reason}"
        }
    
    def _get_default_signal(self):
        """Return default HOLD signal if API fails"""
        return {
            "signal": "HOLD",
            "stop_price": None,
            "target_price": None,
            "confidence": 50,
            "reason": "API request failed, defaulting to HOLD"
        }
    
    def evaluate_trade_profitability(self, signal, entry_price, stop_price, target_price, 
                                   future_prices, hours_to_evaluate=24):
        """
        Evaluate if the trade would have been profitable
        """
        if signal == "HOLD" or not future_prices:
            return False, "HOLD", 0
        
        # Set default stop and target if not provided
        if stop_price is None:
            stop_price = entry_price * 0.98 if signal == "BUY" else entry_price * 1.02
        
        if target_price is None:
            target_price = entry_price * 1.03 if signal == "BUY" else entry_price * 0.97
        
        # Check if stop or target hit within next X hours
        for i, future_price in enumerate(future_prices[:hours_to_evaluate]):
            if signal == "BUY":
                if future_price <= stop_price:
                    return False, "STOP_LOSS", ((stop_price - entry_price) / entry_price) * 100
                if future_price >= target_price:
                    return True, "TAKE_PROFIT", ((target_price - entry_price) / entry_price) * 100
            
            elif signal == "SELL":
                if future_price >= stop_price:
                    return False, "STOP_LOSS", ((entry_price - stop_price) / entry_price) * 100
                if future_price <= target_price:
                    return True, "TAKE_PROFIT", ((entry_price - target_price) / entry_price) * 100
        
        # If neither hit, use final price
        final_price = future_prices[min(hours_to_evaluate, len(future_prices)) - 1]
        if signal == "BUY":
            pnl = ((final_price - entry_price) / entry_price) * 100
            outcome = "EXIT_AT_END"
        else:  # SELL
            pnl = ((entry_price - final_price) / entry_price) * 100
            outcome = "EXIT_AT_END"
        
        return pnl > 0, outcome, pnl

