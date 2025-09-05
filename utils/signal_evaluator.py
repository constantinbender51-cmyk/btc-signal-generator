import json
import aiohttp
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class SignalEvaluator:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY', '')
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
    def format_ohlc_data(self, df_chunk):
        """Format OHLC data for the prompt"""
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
        return formatted_data
    
    async def generate_signal(self, ohlc_data: list) -> Dict[str, Any]:
        """Generate trading signal using DeepSeek API"""
        
        prompt = f"""
        Analyze the following BTC/USDT hourly OHLC data and generate a trading signal.
        Respond with a JSON object containing: signal (BUY|SELL|HOLD), stop_price, target_price, 
        confidence (0-100), and reason.
        
        OHLC Data (most recent last):
        {json.dumps(ohlc_data, indent=2)}
        
        Important: Consider technical analysis, price action, volume patterns, and market structure.
        Provide realistic stop and target prices based on support/resistance levels.
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
                    "content": "You are an expert cryptocurrency trading analyst. Analyze OHLC data and provide clear trading signals with proper risk management."
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
                            # Find JSON object in the response
                            json_start = response_content.find('{')
                            json_end = response_content.rfind('}') + 1
                            if json_start != -1 and json_end != -1:
                                json_str = response_content[json_start:json_end]
                                signal_data = json.loads(json_str)
                                return signal_data
                        except json.JSONDecodeError:
                            print("Failed to parse JSON response")
                            return self._get_default_signal()
                    
                    print(f"API request failed with status {response.status}")
                    return self._get_default_signal()
                    
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return self._get_default_signal()
    
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
        Returns: (is_profitable, actual_outcome, profit_loss_percent)
        """
        if signal == "HOLD":
            return False, "HOLD", 0
        
        # Check if stop or target hit within next X hours
        for i, future_price in enumerate(future_prices[:hours_to_evaluate]):
            if signal == "BUY":
                if stop_price and future_price <= stop_price:
                    return False, "STOP_LOSS", ((stop_price - entry_price) / entry_price) * 100
                if target_price and future_price >= target_price:
                    return True, "TAKE_PROFIT", ((target_price - entry_price) / entry_price) * 100
            
            elif signal == "SELL":
                if stop_price and future_price >= stop_price:
                    return False, "STOP_LOSS", ((entry_price - stop_price) / entry_price) * 100
                if target_price and future_price <= target_price:
                    return True, "TAKE_PROFIT", ((entry_price - target_price) / entry_price) * 100
        
        # If neither hit, use final price
        final_price = future_prices[hours_to_evaluate - 1]
        if signal == "BUY":
            pnl = ((final_price - entry_price) / entry_price) * 100
            outcome = "EXIT_AT_END"
        else:  # SELL
            pnl = ((entry_price - final_price) / entry_price) * 100
            outcome = "EXIT_AT_END"
        
        return pnl > 0, outcome, pnl
