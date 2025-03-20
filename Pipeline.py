# Pipeline.py

import pandas as pd
import numpy as np
import yfinance as yf

def fetch_stock_data(ticker='RIVN', start_date='2022-01-01', end_date='2024-01-01'):
    """
    Fetch historical stock data from Yahoo Finance.
    Default ticker is set to Rivian (RIVN) for this project.
    """
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Data fetched: {len(data)} rows")
    return data

def add_features(data):
    """
    Add technical indicators and lag features to the stock data.
    """
    df = data.copy()
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change()

    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

    # Lag Features - last 5 days Close prices
    for lag in range(1, 6):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

    # Drop rows with NaN values created due to rolling calculations
    df.dropna(inplace=True)
    
    print("Feature engineering completed.")
    return df

if __name__ == "__main__":
    # Example Run
    stock_data = fetch_stock_data()
    feature_data = add_features(stock_data)
    print(feature_data.head())
