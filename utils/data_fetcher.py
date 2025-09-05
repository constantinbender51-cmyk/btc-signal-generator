import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

class BTCDataFetcher:
    def __init__(self):
        try:
            self.exchange = ccxt.binance({
                'timeout': 30000,
                'enableRateLimit': True,
                'rateLimit': 1000,
            })
            # Test connection
            self.exchange.load_markets()
            logger.info("Successfully connected to Binance API")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            # Fallback to a simpler approach
            self.exchange = None
    
    def fetch_historical_data(self, years=1):
        """Fetch historical BTC/USDT data with fallback"""
        logger.info(f"Fetching {years} years of BTC data...")
        
        try:
            # Try to fetch from exchange
            if self.exchange:
                since = int((datetime.now() - timedelta(days=365*years)).timestamp() * 1000)
                all_ohlcv = []
                
                # Fetch data in chunks
                while len(all_ohlcv) < 365 * years * 24:
                    try:
                        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=1000)
                        if not ohlcv:
                            break
                        
                        since = ohlcv[-1][0] + 1
                        all_ohlcv.extend(ohlcv)
                        
                        # Respect rate limits
                        time.sleep(self.exchange.rateLimit / 1000)
                        
                    except Exception as e:
                        logger.error(f"Error fetching chunk: {e}")
                        break
                
                if all_ohlcv:
                    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    logger.info(f"Successfully fetched {len(df)} hourly candles from exchange")
                    return df
            
            # Fallback: Generate synthetic data or use cached data
            logger.warning("Using fallback synthetic data")
            return self._generate_fallback_data(years)
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return self._generate_fallback_data(years)
    
    def _generate_fallback_data(self, years=1):
        """Generate fallback synthetic data"""
        start_date = datetime.now() - timedelta(days=365*years)
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='H')
        
        # Create synthetic price data that resembles BTC
        np.random.seed(42)
        base_price = 30000
        returns = np.random.normal(0.0001, 0.01, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add some volatility
        volatility = 0.02
        highs = prices * (1 + np.random.uniform(0, volatility, len(dates)))
        lows = prices * (1 - np.random.uniform(0, volatility, len(dates)))
        
        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates))
        }, index=dates)
        
        logger.info(f"Generated {len(df)} synthetic hourly candles")
        return df

    def get_data_chunk(self, df, start_index, chunk_size=50):
        """Get a chunk of OHLC data for analysis"""
        if df is None or start_index + chunk_size > len(df):
            return None
        
        chunk = df.iloc[start_index:start_index + chunk_size].copy()
        return chunk

async def fetch_current_price():
    """Fetch current BTC price with fallback"""
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker('BTC/USDT')
        return ticker['last']
    except Exception as e:
        logger.error(f"Error fetching current price: {e}")
        return 40000  # Fallback price
