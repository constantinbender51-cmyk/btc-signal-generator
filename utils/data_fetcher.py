import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import asyncio

class BTCDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance()
        
    def fetch_historical_data(self, years=10):
        """Fetch 10 years of hourly BTC/USDT data"""
        since = int((datetime.now() - timedelta(days=365*years)).timestamp() * 1000)
        all_ohlcv = []
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=1000)
                if not ohlcv:
                    break
                
                since = ohlcv[-1][0] + 1
                all_ohlcv.extend(ohlcv)
                
                # Respect rate limits
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df

    def get_data_chunk(self, df, start_index, chunk_size=50):
        """Get a chunk of OHLC data for analysis"""
        if start_index + chunk_size > len(df):
            return None
        
        chunk = df.iloc[start_index:start_index + chunk_size].copy()
        return chunk

async def fetch_current_price():
    """Fetch current BTC price"""
    exchange = ccxt.binance()
    try:
        ticker = exchange.fetch_ticker('BTC/USDT')
        return ticker['last']
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None
