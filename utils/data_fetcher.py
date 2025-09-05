import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import asyncio

class BTCDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binusdm()  # Using Binance USDM futures for more data
        
    def fetch_historical_data(self, years=2):
        """Fetch historical BTC/USDT data"""
        print(f"Fetching {years} years of BTC data...")
        
        # Calculate start time (reduce to 2 years for faster loading)
        since = int((datetime.now() - timedelta(days=365*years)).timestamp() * 1000)
        all_ohlcv = []
        
        # Fetch data in chunks
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=1000)
                if not ohlcv:
                    break
                
                since = ohlcv[-1][0] + 1
                all_ohlcv.extend(ohlcv)
                
                # Break if we have enough data (approx 2 years)
                if len(all_ohlcv) >= 365 * 2 * 24:  # 2 years of hourly data
                    break
                    
                # Respect rate limits
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        if not all_ohlcv:
            raise Exception("Failed to fetch historical data")
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Successfully fetched {len(df)} hourly candles")
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
