import yfinance as yf
import pandas as pd
from datetime import datetime

def extracting_features(df):
    """Extracts key financial indicators from a Yahoo Finance dataset."""
    
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()

    # Feature 1: Price Change (Close - Open)
    df['price_change'] = df['Close'] - df['Open']
    print("Added feature: 'price_change' (Close - Open)")

    # Feature 2: Returns (Daily Percentage Change)
    df['returns'] = df['Close'].pct_change()
    print("Added feature: 'returns' (Daily Percentage Change)")
    
    # Feature 3: Average Price (Average of Open and Close)
    df['average_price'] = (df['Close'] + df['Open']) / 2
    print("Added feature: 'average_price' (Average of Open and Close)")
    
    # Feature 4: Price Range (High - Low)
    df['price_range'] = df['High'] - df['Low']
    print("Added feature: 'price_range' (High - Low)")
    
    # Feature 5: Volume Change (Difference in Volume)
    df['volume_change'] = df['Volume'].diff()
    print("Added feature: 'volume_change' (Difference in Volume)")
    
    # Feature 6: Moving Average (10-Day)
    df['moving_average_10'] = df['Close'].rolling(window=10).mean()
    print("Added feature: 'moving_average_10' (10-Day Moving Average)")

    # Feature 7: Relative Strength Index (RSI) - 14-Day window
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    print("Added feature: 'RSI' (14-Day Relative Strength Index)")

    # Feature 8: MACD (Moving Average Convergence Divergence)
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()  # Short-term EMA (12-day)
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()   # Long-term EMA (26-day)
    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line (9-day EMA)
    print("Added feature: 'MACD' (Moving Average Convergence Divergence) and 'MACD_Signal'")
    
    # Feature 9: Bollinger Bands (20-Day Moving Average ± 2 Standard Deviations)
    df['20_SMA'] = df['Close'].rolling(window=20).mean()  # 20-Day Simple Moving Average
    df['BB_Upper'] = df['20_SMA'] + (df['Close'].rolling(window=20).std() * 2)  # Upper Bollinger Band
    df['BB_Lower'] = df['20_SMA'] - (df['Close'].rolling(window=20).std() * 2)  # Lower Bollinger Band
    print("Added feature: '20_SMA' (20-Day Moving Average), 'BB_Upper' (Upper Bollinger Band), and 'BB_Lower' (Lower Bollinger Band)")

    # Return the DataFrame with the newly added features
    return df


def download_and_process_data(ticker='AAPL'):
    """Downloads and processes the stock data for the given ticker."""
    print(f'Extracting New Features for {ticker}')
    print('--------------------------------------')
    
    # Download the data for the given stock ticker
    df = yf.download(ticker, start='2020-01-01', end=datetime.now(), auto_adjust=False)
    
    # Reshape the DataFrame so that tickers appear in rows (in case of multi-level columns)
    df.columns = df.columns.droplevel(1)

    # Filter only the relevant columns (Price, Adj Close, Close, High, Low, Open, Volume)
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    # Apply the feature extraction function
    df = extracting_features(df)
    print(df.tail(5))
    print(f'New features have been successfully added to the DataFrame for {ticker}.')
    print('-----------------------------------------------------------')
    
    return df
