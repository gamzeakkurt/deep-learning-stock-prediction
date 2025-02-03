# Essential Libraries

# Data handling and statistical analysis
import pandas as pd
import numpy as np
from scipy import stats
#import pandas_datareader as pdr  # Ensure correct import

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Max-min scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Machine Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.linear_model import LinearRegression

# Optimization and allocation
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt import black_litterman, BlackLittermanModel

# Financial data
import quantstats as qs
import ta
import yfinance as yf

# For time stamps
from datetime import datetime
import datetime as dt
import warnings
import pytz
import os

# Other
from tabulate import tabulate

# Enabling Plotly offline
from plotly.offline import init_notebook_mode
import plotly.io as pio
pio.renderers.default = "colab"

# Ignore warnings
warnings.filterwarnings("ignore")

"""
We will use Yahoo Finance to analyze the stock market performance of four major technology companies: 
Apple (AAPL), Amazon (AMZN), Tesla (TSLA), and Microsoft (MSFT). 
Our dataset contains daily returns from January 1, 2020, to January 1, 2025.
"""

# data_acquisition.py

warnings.filterwarnings("ignore")

def load_stock_data(tickers, start_date='2020-01-01', end_date='2025-01-01'):
    """
    Downloads stock data for the given tickers and returns a cleaned DataFrame.
    """
    stocks_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
    
    # Reshape the DataFrame so tickers appear in rows
    df = stocks_data.stack(level=0).reset_index()
    
    # Rename columns for clarity
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Set date index
    df = df.set_index('Date')
    
    return df

if __name__ == "__main__":
    tickers = ['AAPL', 'AMZN', 'TSLA', 'MSFT']
    df = load_stock_data(tickers)
    print(df.tail(10))  # Only runs if executed directly



